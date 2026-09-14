//! The OpenAI-compatible protocol: configuration, request shaping, SSE
//! streaming, error handling.
//!
//! Any service that speaks OpenAI Chat Completions works — OpenAI, DeepSeek,
//! OpenRouter, Qwen, GLM, Moonshot, Groq, Together, Fireworks, vLLM, Ollama,
//! LM Studio, proxies — by pointing [`OpenAiConfig::base_url`] at it. No vendor
//! is special-cased, and provider-specific fields go through
//! [`OpenAiConfig::param`] instead of being enumerated here.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::{Event, Message, Model, Response, ToolCall, ToolDefinition, clip};

/// Body fields the protocol owns. `.param()` may not override these, or the
/// crate's own assumptions about the request would quietly stop holding.
const RESERVED: [&str; 4] = ["model", "messages", "tools", "stream"];

pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
pub const DEFAULT_MODEL: &str = "gpt-4o-mini";
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(600);
/// Conventional environment variable holding the key.
pub const API_KEY_ENV: &str = "OPENAI_API_KEY";

const ERROR_BODY_LIMIT: usize = 2000;

/// Everything the wire protocol needs. Build it with the chainable setters:
///
/// ```no_run
/// # use agent::OpenAiConfig;
/// let config = OpenAiConfig::default()
///     .model("deepseek-chat")
///     .base_url("https://api.deepseek.com")
///     .api_key("sk-…")
///     .header("x-tenant", "acme")
///     .param("reasoning_effort", "high")
///     .param("temperature", 0.2);
/// ```
#[derive(Debug, Clone)]
pub struct OpenAiConfig {
    pub model: String,
    pub base_url: String,
    /// `None` for endpoints that need no key, such as a local Ollama or vLLM.
    pub api_key: Option<String>,
    pub timeout: Duration,
    /// Extra request headers, applied last.
    pub headers: Vec<(String, String)>,
    /// Provider-specific request fields, merged into the body as given. This is
    /// the escape hatch for anything this crate does not name: `temperature`,
    /// `top_p`, `max_completion_tokens`, `seed`, `tool_choice`, `response_format`,
    /// `reasoning_effort`, `parallel_tool_calls`, ...
    pub extra_body: Map<String, Value>,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_MODEL.to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: None,
            timeout: DEFAULT_TIMEOUT,
            headers: Vec::new(),
            extra_body: Map::new(),
        }
    }
}

impl OpenAiConfig {
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Add or replace one field of the request body.
    pub fn param(mut self, name: impl Into<String>, value: impl Into<Value>) -> Self {
        self.extra_body.insert(name.into(), value.into());
        self
    }
}

pub struct OpenAiCompatible {
    config: OpenAiConfig,
    client: reqwest::Client,
}

impl OpenAiCompatible {
    /// Start from [`OpenAiConfig::default`].
    pub fn builder() -> OpenAiConfig {
        OpenAiConfig::default()
    }

    pub fn new(config: OpenAiConfig) -> Result<Self> {
        for reserved in RESERVED {
            if config.extra_body.contains_key(reserved) {
                bail!("`{reserved}` belongs to the protocol and cannot be set with .param()");
            }
        }
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .context("failed to build the HTTP client")?;
        let config = OpenAiConfig {
            base_url: config.base_url.trim_end_matches('/').to_string(),
            ..config
        };
        Ok(Self { config, client })
    }

    pub fn config(&self) -> &OpenAiConfig {
        &self.config
    }

    async fn send(&self, body: &Value) -> Result<reqwest::Response> {
        let url = format!("{}/chat/completions", self.config.base_url);
        let mut request = self.client.post(&url).json(body);
        if let Some(key) = &self.config.api_key {
            request = request.bearer_auth(key);
        }
        for (name, value) in &self.config.headers {
            request = request.header(name, value);
        }
        let response = request
            .send()
            .await
            .with_context(|| format!("request to {url} failed"))?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            bail!("api error {status}: {}", clip(&text, ERROR_BODY_LIMIT));
        }
        Ok(response)
    }
}

impl Model for OpenAiCompatible {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        on_event: &mut dyn FnMut(Event<'_>),
    ) -> Result<Response> {
        let mut response = self.send(&body(&self.config, messages, tools)).await?;

        if !is_event_stream(&response) {
            bail!(
                "the endpoint did not stream (content-type: {}): \
                 it may not support `stream`",
                response
                    .headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .and_then(|value| value.to_str().ok())
                    .unwrap_or("none"),
            );
        }

        let mut stream = Stream::default();
        read_sse(&mut response, |payload| stream.apply(payload, on_event)).await?;
        let response = stream.finish();
        if response.text.is_none() && response.tool_calls.is_empty() {
            bail!("the model returned neither text nor tool calls");
        }
        Ok(response)
    }
}

fn is_event_stream(response: &reqwest::Response) -> bool {
    response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.contains("event-stream"))
}

/// Read an SSE body, handing every `data:` payload to `on_event`.
async fn read_sse(
    response: &mut reqwest::Response,
    mut on_event: impl FnMut(&str) -> Result<()>,
) -> Result<()> {
    let mut buffer = String::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .context("failed to read the stream")?
    {
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(end) = buffer.find('\n') {
            let line = buffer[..end].trim_end_matches('\r').to_string();
            buffer.drain(..=end);
            if let Some(payload) = line.strip_prefix("data:") {
                let payload = payload.trim();
                if !payload.is_empty() && payload != "[DONE]" {
                    on_event(payload)?;
                }
            }
        }
    }
    Ok(())
}

// ------------------------------------------------------------------- request

fn body(config: &OpenAiConfig, messages: &[Message], tools: &[ToolDefinition]) -> Value {
    let mut msgs = Vec::with_capacity(messages.len());
    for message in messages {
        match message {
            Message::System(text) => msgs.push(json!({"role": "system", "content": text})),
            Message::User(text) => msgs.push(json!({"role": "user", "content": text})),
            Message::Assistant {
                text,
                calls,
                metadata,
            } => {
                let mut msg = json!({"role": "assistant"});
                // An assistant turn with tool calls must carry them in the same
                // message, so `content` is explicitly null when there is no text.
                msg["content"] = if text.is_empty() {
                    Value::Null
                } else {
                    json!(text)
                };
                if !calls.is_empty() {
                    msg["tool_calls"] = Value::Array(
                        calls
                            .iter()
                            .map(|call| {
                                json!({
                                    "id": call.id,
                                    "type": "function",
                                    "function": {
                                        "name": call.name,
                                        "arguments": call.arguments.to_string(),
                                    }
                                })
                            })
                            .collect(),
                    );
                }
                if let Some(object) = msg.as_object_mut() {
                    // Echo back the provider fields captured from its own reply.
                    for (name, value) in metadata {
                        object.insert(name.clone(), value.clone());
                    }
                }
                msgs.push(msg);
            }
            // One `tool` message per result.
            Message::Tools(results) => {
                for result in results {
                    msgs.push(json!({
                        "role": "tool",
                        "tool_call_id": result.id,
                        "content": result.output,
                    }));
                }
            }
        }
    }

    let mut body = json!({
        "model": config.model,
        "messages": msgs,
        "stream": true,
    });
    if !tools.is_empty() {
        body["tools"] = Value::Array(
            tools
                .iter()
                .map(|tool| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        }
                    })
                })
                .collect(),
        );
    }
    // Provider-specific fields come last, so a host can override anything above.
    if let Some(object) = body.as_object_mut() {
        for (name, value) in &config.extra_body {
            object.insert(name.clone(), value.clone());
        }
    }
    body
}

// ------------------------------------------------------------------ response

#[derive(Deserialize)]
struct Chunk {
    #[serde(default)]
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    delta: ChunkDelta,
}

#[derive(Deserialize)]
struct ChunkDelta {
    #[serde(default)]
    content: Option<String>,
    /// Reasoning models stream their thinking next to the answer.
    #[serde(default)]
    reasoning: Option<String>,
    /// Fields a provider expects echoed back on the next request.
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    reasoning_details: Option<Value>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Deserialize)]
struct ToolCallDelta {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<FunctionDelta>,
}

#[derive(Deserialize)]
struct FunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// A tool call under construction: its arguments arrive in fragments.
#[derive(Default)]
struct CallBuilder {
    id: String,
    name: String,
    arguments: String,
}

impl CallBuilder {
    fn finish(self) -> ToolCall {
        ToolCall {
            id: self.id,
            name: self.name,
            arguments: if self.arguments.trim().is_empty() {
                json!({})
            } else {
                // Unparseable arguments are kept verbatim, as a JSON string, so
                // the tool reports what the model actually sent.
                match serde_json::from_str(&self.arguments) {
                    Ok(arguments) => arguments,
                    Err(_) => Value::String(self.arguments),
                }
            },
        }
    }
}

/// The state of one streamed turn.
#[derive(Default)]
struct Stream {
    text: String,
    calls: Vec<CallBuilder>,
    /// Round-trip fields, collected from the stream and handed back to the agent.
    reasoning_content: String,
    reasoning_details: Option<Value>,
}

impl Stream {
    fn apply(&mut self, payload: &str, on_event: &mut dyn FnMut(Event<'_>)) -> Result<()> {
        let chunk: Chunk = serde_json::from_str(payload).context("unexpected stream event")?;
        for choice in chunk.choices {
            if let Some(text) = choice.delta.content.filter(|text| !text.is_empty()) {
                self.text.push_str(&text);
                on_event(Event::Text(&text));
            }
            if let Some(thinking) = choice.delta.reasoning.filter(|text| !text.is_empty()) {
                on_event(Event::Thinking(&thinking));
            }
            if let Some(content) = choice.delta.reasoning_content {
                self.reasoning_content.push_str(&content);
            }
            if let Some(details) = choice.delta.reasoning_details {
                self.reasoning_details = Some(details);
            }
            for call in choice.delta.tool_calls.unwrap_or_default() {
                while self.calls.len() <= call.index {
                    self.calls.push(CallBuilder::default());
                }
                let builder = &mut self.calls[call.index];
                if let Some(id) = call.id {
                    builder.id = id;
                }
                if let Some(function) = call.function {
                    if let Some(name) = function.name {
                        builder.name = name;
                    }
                    if let Some(arguments) = function.arguments {
                        builder.arguments.push_str(&arguments);
                    }
                }
            }
        }
        Ok(())
    }

    fn finish(self) -> Response {
        let mut metadata = Map::new();
        if !self.reasoning_content.is_empty() {
            metadata.insert("reasoning_content".into(), json!(self.reasoning_content));
        }
        if let Some(details) = self.reasoning_details {
            metadata.insert("reasoning_details".into(), details);
        }
        Response {
            text: (!self.text.is_empty()).then_some(self.text),
            tool_calls: self.calls.into_iter().map(CallBuilder::finish).collect(),
            metadata,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    /// Records the events a stream emits, in order.
    #[derive(Default)]
    struct Sink(Vec<String>);

    impl Sink {
        fn callback(&mut self) -> impl FnMut(Event<'_>) + '_ {
            |event| {
                self.0.push(match event {
                    Event::Text(text) => format!("text:{text}"),
                    Event::Thinking(text) => format!("thinking:{text}"),
                    other => format!("other:{other:?}"),
                })
            }
        }
    }

    fn config() -> OpenAiConfig {
        OpenAiConfig::default().model("test-model")
    }

    #[test]
    fn streams_text_and_reasoning() {
        let mut stream = Stream::default();
        let mut sink = Sink::default();
        let mut callback = sink.callback();
        for payload in [
            r#"{"choices":[{"delta":{"role":"assistant"}}]}"#,
            r#"{"choices":[{"delta":{"reasoning":"hmm"}}]}"#,
            r#"{"choices":[{"delta":{"content":"Hel"}}]}"#,
            r#"{"choices":[{"delta":{"content":"lo"}}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
        ] {
            stream.apply(payload, &mut callback).unwrap();
        }
        drop(callback);
        let response = stream.finish();

        assert_eq!(response.text.as_deref(), Some("Hello"));
        assert_eq!(sink.0, ["thinking:hmm", "text:Hel", "text:lo"]);
        assert!(response.tool_calls.is_empty());
    }

    #[test]
    fn assembles_fragmented_tool_calls() {
        let mut stream = Stream::default();
        let mut sink = Sink::default();
        let mut callback = sink.callback();
        for payload in [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"read","arguments":"{\"pa"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"a.rs\"}"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_2","function":{"name":"exec","arguments":"{\"command\":\"ls\"}"}}]}}]}"#,
        ] {
            stream.apply(payload, &mut callback).unwrap();
        }
        drop(callback);
        let response = stream.finish();

        assert!(response.text.is_none());
        assert_eq!(response.tool_calls.len(), 2);
        assert_eq!(response.tool_calls[0].id, "call_1");
        assert_eq!(response.tool_calls[0].name, "read");
        assert_eq!(response.tool_calls[0].arguments, json!({"path": "a.rs"}));
        assert_eq!(response.tool_calls[1].name, "exec");
        assert_eq!(response.tool_calls[1].arguments, json!({"command": "ls"}));
    }

    #[test]
    fn keeps_unparseable_arguments_verbatim() {
        let mut stream = Stream::default();
        let mut sink = Sink::default();
        let mut callback = sink.callback();
        stream
            .apply(
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"read","arguments":"{\"path\":"}}]}}]}"#,
                &mut callback,
            )
            .unwrap();
        drop(callback);

        let arguments = &stream.finish().tool_calls[0].arguments;
        assert_eq!(arguments.as_str(), Some("{\"path\":"));
    }

    #[test]
    fn builds_a_request_with_history_tools_and_params() {
        let messages = vec![
            Message::System("sys".into()),
            Message::User("hi".into()),
            Message::Assistant {
                text: String::new(),
                calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "read".into(),
                    arguments: json!({"path": "a.rs"}),
                }],
                metadata: Map::new(),
            },
            Message::Tools(vec![crate::ToolResult {
                id: "call_1".into(),
                name: "read".into(),
                output: "fn main() {}".into(),
            }]),
        ];
        let tools = vec![ToolDefinition {
            name: "read".into(),
            description: "read a file".into(),
            parameters: json!({"type": "object"}),
        }];
        let config = config()
            .param("temperature", 0.2)
            .param("reasoning_effort", "high");
        let body = body(&config, &messages, &tools);

        assert_eq!(body["stream"], true);
        assert_eq!(body["messages"][0]["role"], "system");
        // An assistant turn with calls has null content and carries the calls.
        assert!(body["messages"][2]["content"].is_null());
        assert_eq!(
            body["messages"][2]["tool_calls"][0]["function"]["arguments"],
            "{\"path\":\"a.rs\"}"
        );
        assert_eq!(body["messages"][3]["role"], "tool");
        assert_eq!(body["messages"][3]["tool_call_id"], "call_1");
        assert_eq!(body["tools"][0]["function"]["name"], "read");
        assert_eq!(body["temperature"], 0.2);
        assert_eq!(body["reasoning_effort"], "high");
        assert_eq!(body["model"], "test-model");
    }

    #[test]
    fn rejects_params_the_protocol_owns() {
        for reserved in RESERVED {
            let error = match OpenAiCompatible::new(config().param(reserved, "x")) {
                Err(error) => error.to_string(),
                Ok(_) => panic!("{reserved} should be rejected"),
            };
            assert!(error.contains(reserved), "{error}");
        }
        // Everything else still goes through.
        assert!(OpenAiCompatible::new(config().param("seed", 7)).is_ok());
    }

    #[test]
    fn round_trips_reasoning_metadata() {
        let mut stream = Stream::default();
        let mut sink = Sink::default();
        let mut callback = sink.callback();
        for payload in [
            r#"{"choices":[{"delta":{"reasoning_content":"We "}}]}"#,
            r#"{"choices":[{"delta":{"reasoning_content":"think."}}]}"#,
            r#"{"choices":[{"delta":{"reasoning_details":[{"type":"reasoning.text","text":"We think."}]}}]}"#,
            r#"{"choices":[{"delta":{"content":"done"}}]}"#,
        ] {
            stream.apply(payload, &mut callback).unwrap();
        }
        drop(callback);
        let response = stream.finish();

        assert_eq!(
            response.metadata.get("reasoning_content"),
            Some(&json!("We think."))
        );
        let details = response.metadata.get("reasoning_details").unwrap();
        assert_eq!(details[0]["type"], "reasoning.text");

        // ... and the same map comes back on the next request.
        let mut messages = vec![Message::User("hi".into())];
        messages.push(Message::Assistant {
            text: response.text.clone().unwrap_or_default(),
            calls: response.tool_calls.clone(),
            metadata: response.metadata.clone(),
        });
        let body = body(&config(), &messages, &[]);
        assert_eq!(body["messages"][1]["reasoning_content"], "We think.");
        assert_eq!(
            body["messages"][1]["reasoning_details"][0]["type"],
            "reasoning.text"
        );
        // Fields that do not need round-tripping are not echoed.
        assert!(body["messages"][1].get("content").is_some());
    }

    // ------------------------------------------------------------- over HTTP

    /// A throwaway OpenAI-compatible server: `handler` sees the raw request and
    /// returns the raw response, so a test can encode its expectations in the
    /// reply and let a failed expectation surface as a request error.
    async fn serve(handler: impl Fn(&str) -> String + Send + 'static) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let request = read_request(&mut socket).await;
                let _ = socket.write_all(handler(&request).as_bytes()).await;
                let _ = socket.shutdown().await;
                let mut sink = [0u8; 1024];
                let _ = tokio::time::timeout(Duration::from_secs(5), socket.read(&mut sink)).await;
            }
        });
        format!("http://{address}")
    }

    async fn read_request(socket: &mut TcpStream) -> String {
        let mut request = Vec::new();
        let mut chunk = [0u8; 4096];
        while let Ok(read) = socket.read(&mut chunk).await {
            if read == 0 {
                break;
            }
            request.extend_from_slice(&chunk[..read]);
            let Some(head_end) = find(&request, b"\r\n\r\n") else {
                continue;
            };
            let head = String::from_utf8_lossy(&request[..head_end]).to_lowercase();
            let length = head
                .lines()
                .find_map(|line| line.strip_prefix("content-length:"))
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if request.len() >= head_end + 4 + length {
                break;
            }
        }
        String::from_utf8_lossy(&request).to_string()
    }

    fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack.windows(needle.len()).position(|w| w == needle)
    }

    fn http(status: &str, content_type: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    fn sse(events: &[&str]) -> String {
        http(
            "200 OK",
            "text/event-stream",
            &events
                .iter()
                .map(|event| format!("data: {event}\n\n"))
                .collect::<String>(),
        )
    }

    #[tokio::test]
    async fn sends_params_and_headers_and_reads_the_stream() {
        let base_url = serve(|request| {
            let head = request.to_lowercase();
            let checks = [
                (
                    head.contains("authorization: bearer sekret"),
                    "no auth header",
                ),
                (head.contains("x-tenant: acme"), "no custom header"),
                (
                    request.contains("\"temperature\":0.2"),
                    "no temperature param",
                ),
                (
                    request.contains("\"max_tokens\":128"),
                    "no max_tokens param",
                ),
                (request.contains("\"stream\":true"), "stream not requested"),
            ];
            match checks.iter().find(|(ok, _)| !ok) {
                Some((_, why)) => http("400 Bad Request", "application/json", why),
                None => sse(&[r#"{"choices":[{"delta":{"content":"done"}}]}"#]),
            }
        })
        .await;

        let model = OpenAiCompatible::new(
            config()
                .base_url(base_url)
                .api_key("sekret")
                .header("x-tenant", "acme")
                .param("temperature", 0.2)
                .param("max_tokens", 128),
        )
        .unwrap();

        let mut text = String::new();
        let response = model
            .chat(&[Message::User("hi".into())], &[], &mut |event| {
                if let Event::Text(chunk) = event {
                    text.push_str(chunk);
                }
            })
            .await
            .unwrap();

        assert_eq!(response.text.as_deref(), Some("done"));
        assert_eq!(text, "done");
    }

    #[tokio::test]
    async fn works_without_an_api_key() {
        let base_url = serve(|request| {
            if request.to_lowercase().contains("authorization") {
                return http("400 Bad Request", "application/json", "unexpected auth");
            }
            sse(&[r#"{"choices":[{"delta":{"content":"ok"}}]}"#])
        })
        .await;

        let model = OpenAiCompatible::new(config().base_url(base_url)).unwrap();
        let response = model
            .chat(&[Message::User("hi".into())], &[], &mut |_| {})
            .await
            .unwrap();

        assert_eq!(response.text.as_deref(), Some("ok"));
    }

    #[tokio::test]
    async fn surfaces_the_error_body() {
        let base_url = serve(|_| {
            http(
                "401 Unauthorized",
                "application/json",
                r#"{"error":{"message":"bad key"}}"#,
            )
        })
        .await;

        let model = OpenAiCompatible::new(config().base_url(base_url)).unwrap();
        let error = model
            .chat(&[Message::User("hi".into())], &[], &mut |_| {})
            .await
            .unwrap_err();

        let message = format!("{error:#}");
        assert!(message.contains("401"), "{message}");
        assert!(message.contains("bad key"), "{message}");
    }
}
