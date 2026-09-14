//! The only place that knows about API protocols.
//!
//! Everything above this module speaks [`Message`] / [`Response`]. Each protocol
//! is one `*_body` builder plus one streaming accumulator, and a call is a
//! `match`. There is no provider registry and no whole-body parsing path: every
//! protocol we support can stream, so that is the only path there is.

use std::fmt;
use std::str::FromStr;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::tools::{ToolDefinition, clip};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
const ERROR_BODY_LIMIT: usize = 2000;
const ANTHROPIC_MAX_TOKENS: u32 = 4096;

/// The API protocol, not the vendor: every service that speaks OpenAI Chat
/// Completions (OpenAI, DeepSeek, OpenRouter, Qwen, GLM, Moonshot, Groq,
/// Together, Fireworks, vLLM, Ollama, LM Studio, ...) is [`Api::OpenAi`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Api {
    OpenAi,
    Anthropic,
    Gemini,
}

impl Api {
    pub fn default_base_url(self) -> &'static str {
        match self {
            Api::OpenAi => "https://api.openai.com/v1",
            Api::Anthropic => "https://api.anthropic.com",
            Api::Gemini => "https://generativelanguage.googleapis.com",
        }
    }

    pub fn default_model(self) -> &'static str {
        match self {
            Api::OpenAi => "gpt-4o-mini",
            Api::Anthropic => "claude-sonnet-4-5",
            Api::Gemini => "gemini-2.5-flash",
        }
    }

    /// Conventional environment variables holding a key for this protocol.
    pub fn api_key_envs(self) -> &'static [&'static str] {
        match self {
            Api::OpenAi => &["OPENAI_API_KEY"],
            Api::Anthropic => &["ANTHROPIC_API_KEY"],
            Api::Gemini => &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        }
    }
}

impl fmt::Display for Api {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Api::OpenAi => "openai",
            Api::Anthropic => "anthropic",
            Api::Gemini => "gemini",
        })
    }
}

impl FromStr for Api {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "openai" | "openai-compatible" | "compatible" => Ok(Api::OpenAi),
            "anthropic" | "claude" => Ok(Api::Anthropic),
            "gemini" | "google" => Ok(Api::Gemini),
            other => bail!("unknown api {other:?}: expected one of openai, anthropic, gemini"),
        }
    }
}

/// One entry of the conversation, in the shape all three protocols need.
///
/// Text and tool calls share one assistant turn, and parallel tool results
/// share one turn: each protocol then maps a variant to one message, so no
/// adapter has to regroup anything.
#[derive(Debug, Clone)]
pub enum Message {
    System(String),
    User(String),
    Assistant { text: String, calls: Vec<ToolCall> },
    Tools(Vec<ToolResult>),
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub id: String,
    pub name: String,
    pub output: String,
}

#[derive(Debug, Clone, Default)]
pub struct Response {
    pub text: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

/// A piece of model output that arrives while a turn is still being generated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delta {
    /// Answer text.
    Text,
    /// Internal reasoning of a reasoning model: worth showing, not part of the answer.
    Thinking,
}

pub struct Llm {
    pub api: Api,
    pub model: String,
    pub base_url: String,
    pub api_key: String,
    client: reqwest::Client,
}

impl Llm {
    /// `model` and `base_url` fall back to the protocol's defaults.
    pub fn new(
        api: Api,
        model: Option<String>,
        base_url: Option<String>,
        api_key: String,
    ) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .context("failed to build the HTTP client")?;
        Ok(Self {
            api,
            model: model.unwrap_or_else(|| api.default_model().to_string()),
            base_url: base_url
                .unwrap_or_else(|| api.default_base_url().to_string())
                .trim_end_matches('/')
                .to_string(),
            api_key,
            client,
        })
    }

    /// Send one turn and stream it, calling `on_delta` as output arrives.
    pub async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        on_delta: &mut dyn FnMut(Delta, &str),
    ) -> Result<Response> {
        let body = match self.api {
            Api::OpenAi => openai_body(&self.model, messages, tools),
            Api::Anthropic => anthropic_body(&self.model, messages, tools),
            Api::Gemini => gemini_body(messages, tools),
        };
        let mut response = self.send(&body).await?;

        if !is_event_stream(&response) {
            bail!(
                "{} did not stream a response (content-type: {}): \
                 this endpoint may not support streaming",
                self.api,
                response
                    .headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .and_then(|value| value.to_str().ok())
                    .unwrap_or("none"),
            );
        }

        let response = match self.api {
            Api::OpenAi => stream_openai(&mut response, on_delta).await?,
            Api::Anthropic => stream_anthropic(&mut response, on_delta).await?,
            Api::Gemini => stream_gemini(&mut response, on_delta).await?,
        };
        if response.text.is_none() && response.tool_calls.is_empty() {
            bail!("{} returned neither text nor tool calls", self.api);
        }
        Ok(response)
    }

    fn url(&self) -> String {
        let base = &self.base_url;
        match self.api {
            Api::OpenAi => format!("{base}/chat/completions"),
            Api::Anthropic if base.ends_with("/v1") => format!("{base}/messages"),
            Api::Anthropic => format!("{base}/v1/messages"),
            Api::Gemini => format!(
                "{base}/v1beta/models/{}:streamGenerateContent?alt=sse",
                self.model
            ),
        }
    }

    /// One POST per turn, with the auth scheme of the protocol, and API errors
    /// surfaced with their response body instead of a bare status code.
    async fn send(&self, body: &Value) -> Result<reqwest::Response> {
        let url = self.url();
        let request = match self.api {
            Api::OpenAi => self.client.post(&url).bearer_auth(&self.api_key),
            Api::Anthropic => self
                .client
                .post(&url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01"),
            Api::Gemini => self
                .client
                .post(&url)
                .header("x-goog-api-key", &self.api_key),
        };
        let response = request
            .json(body)
            .send()
            .await
            .with_context(|| format!("request to {url} failed"))?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            bail!(
                "{} error {status}: {}",
                self.api,
                clip(&text, ERROR_BODY_LIMIT)
            );
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

/// A tool call under construction: OpenAI and Anthropic stream its arguments in
/// fragments, so they are concatenated until the stream ends.
#[derive(Default)]
struct CallBuilder {
    id: String,
    name: String,
    arguments: String,
}

impl CallBuilder {
    /// `None` for a gap: Anthropic numbers every content block, so a stream can
    /// leave an empty slot where a text or thinking block was.
    fn finish(self) -> Option<ToolCall> {
        (!self.name.is_empty()).then(|| ToolCall {
            id: self.id,
            name: self.name,
            arguments: if self.arguments.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(&self.arguments).unwrap_or_else(|_| json!({}))
            },
        })
    }
}

/// The builder for one stream index, created on first sight.
fn slot(calls: &mut Vec<CallBuilder>, index: usize) -> &mut CallBuilder {
    while calls.len() <= index {
        calls.push(CallBuilder::default());
    }
    &mut calls[index]
}

// --------------------------------------------------------------------- openai

fn openai_body(model: &str, messages: &[Message], tools: &[ToolDefinition]) -> Value {
    let mut msgs = Vec::with_capacity(messages.len());
    for message in messages {
        match message {
            Message::System(text) => msgs.push(json!({"role": "system", "content": text})),
            Message::User(text) => msgs.push(json!({"role": "user", "content": text})),
            Message::Assistant { text, calls } => {
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
                msgs.push(msg);
            }
            // OpenAI wants one `tool` message per result.
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

    let mut body = json!({"model": model, "messages": msgs, "stream": true});
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
    body
}

#[derive(Deserialize)]
struct OpenAiChunk {
    #[serde(default)]
    choices: Vec<OpenAiChoice>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    delta: OpenAiDelta,
}

#[derive(Deserialize)]
struct OpenAiDelta {
    #[serde(default)]
    content: Option<String>,
    /// Reasoning models stream their thinking next to the answer.
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiCall>>,
}

#[derive(Deserialize)]
struct OpenAiCall {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OpenAiFunction>,
}

#[derive(Deserialize)]
struct OpenAiFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Default)]
struct OpenAiStream {
    text: String,
    calls: Vec<CallBuilder>,
}

impl OpenAiStream {
    fn apply(&mut self, payload: &str, on_delta: &mut dyn FnMut(Delta, &str)) -> Result<()> {
        let chunk: OpenAiChunk =
            serde_json::from_str(payload).context("unexpected openai stream event")?;
        for choice in chunk.choices {
            if let Some(text) = choice.delta.content.filter(|text| !text.is_empty()) {
                self.text.push_str(&text);
                on_delta(Delta::Text, &text);
            }
            if let Some(thinking) = choice.delta.reasoning.filter(|text| !text.is_empty()) {
                on_delta(Delta::Thinking, &thinking);
            }
            for call in choice.delta.tool_calls.unwrap_or_default() {
                let builder = slot(&mut self.calls, call.index);
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
        Response {
            text: (!self.text.is_empty()).then_some(self.text),
            tool_calls: self
                .calls
                .into_iter()
                .filter_map(CallBuilder::finish)
                .collect(),
        }
    }
}

async fn stream_openai(
    response: &mut reqwest::Response,
    on_delta: &mut dyn FnMut(Delta, &str),
) -> Result<Response> {
    let mut stream = OpenAiStream::default();
    read_sse(response, |payload| stream.apply(payload, on_delta)).await?;
    Ok(stream.finish())
}

// ------------------------------------------------------------------ anthropic

fn anthropic_body(model: &str, messages: &[Message], tools: &[ToolDefinition]) -> Value {
    let mut system = String::new();
    let mut contents = Vec::with_capacity(messages.len());
    for message in messages {
        match message {
            Message::System(text) => {
                if !system.is_empty() {
                    system.push_str("\n\n");
                }
                system.push_str(text);
            }
            Message::User(text) => contents.push(json!({
                "role": "user",
                "content": [{"type": "text", "text": text}],
            })),
            Message::Assistant { text, calls } => {
                let mut blocks = Vec::with_capacity(calls.len() + 1);
                if !text.is_empty() {
                    blocks.push(json!({"type": "text", "text": text}));
                }
                for call in calls {
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }));
                }
                contents.push(json!({"role": "assistant", "content": blocks}));
            }
            // Anthropic wants all results of one turn in a single user message.
            Message::Tools(results) => contents.push(json!({
                "role": "user",
                "content": results
                    .iter()
                    .map(|result| json!({
                        "type": "tool_result",
                        "tool_use_id": result.id,
                        "content": result.output,
                    }))
                    .collect::<Vec<_>>(),
            })),
        }
    }

    let mut body = json!({
        "model": model,
        "max_tokens": ANTHROPIC_MAX_TOKENS,
        "messages": contents,
        "stream": true,
    });
    if !system.is_empty() {
        body["system"] = json!(system);
    }
    if !tools.is_empty() {
        body["tools"] = Value::Array(
            tools
                .iter()
                .map(|tool| {
                    json!({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.parameters,
                    })
                })
                .collect(),
        );
    }
    body
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicEvent {
    ContentBlockStart {
        index: usize,
        content_block: AnthropicStart,
    },
    ContentBlockDelta {
        index: usize,
        delta: AnthropicDelta,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicStart {
    ToolUse {
        id: String,
        name: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicDelta {
    TextDelta {
        text: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    ThinkingDelta {
        thinking: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Default)]
struct AnthropicStream {
    text: String,
    calls: Vec<CallBuilder>,
}

impl AnthropicStream {
    fn apply(&mut self, payload: &str, on_delta: &mut dyn FnMut(Delta, &str)) -> Result<()> {
        let event: AnthropicEvent =
            serde_json::from_str(payload).context("unexpected anthropic stream event")?;
        match event {
            AnthropicEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                if let AnthropicStart::ToolUse { id, name } = content_block {
                    let builder = slot(&mut self.calls, index);
                    builder.id = id;
                    builder.name = name;
                }
            }
            AnthropicEvent::ContentBlockDelta { index, delta } => match delta {
                AnthropicDelta::TextDelta { text } => {
                    self.text.push_str(&text);
                    on_delta(Delta::Text, &text);
                }
                AnthropicDelta::ThinkingDelta { thinking } => {
                    on_delta(Delta::Thinking, &thinking);
                }
                AnthropicDelta::InputJsonDelta { partial_json } => {
                    slot(&mut self.calls, index)
                        .arguments
                        .push_str(&partial_json);
                }
                AnthropicDelta::Other => {}
            },
            AnthropicEvent::Other => {}
        }
        Ok(())
    }

    fn finish(self) -> Response {
        Response {
            text: (!self.text.is_empty()).then_some(self.text),
            tool_calls: self
                .calls
                .into_iter()
                .filter_map(CallBuilder::finish)
                .collect(),
        }
    }
}

async fn stream_anthropic(
    response: &mut reqwest::Response,
    on_delta: &mut dyn FnMut(Delta, &str),
) -> Result<Response> {
    let mut stream = AnthropicStream::default();
    read_sse(response, |payload| stream.apply(payload, on_delta)).await?;
    Ok(stream.finish())
}

// --------------------------------------------------------------------- gemini

fn gemini_body(messages: &[Message], tools: &[ToolDefinition]) -> Value {
    let mut system = String::new();
    let mut contents = Vec::with_capacity(messages.len());
    for message in messages {
        match message {
            Message::System(text) => {
                if !system.is_empty() {
                    system.push_str("\n\n");
                }
                system.push_str(text);
            }
            Message::User(text) => contents.push(json!({
                "role": "user",
                "parts": [{"text": text}],
            })),
            Message::Assistant { text, calls } => {
                let mut parts = Vec::with_capacity(calls.len() + 1);
                if !text.is_empty() {
                    parts.push(json!({"text": text}));
                }
                for call in calls {
                    parts.push(json!({
                        "functionCall": {"name": call.name, "args": call.arguments},
                    }));
                }
                contents.push(json!({"role": "model", "parts": parts}));
            }
            // Gemini names the function instead of the call id.
            Message::Tools(results) => contents.push(json!({
                "role": "user",
                "parts": results
                    .iter()
                    .map(|result| json!({
                        "functionResponse": {
                            "name": result.name,
                            "response": {"output": result.output},
                        }
                    }))
                    .collect::<Vec<_>>(),
            })),
        }
    }

    let mut body = json!({"contents": contents});
    if !system.is_empty() {
        body["systemInstruction"] = json!({"parts": [{"text": system}]});
    }
    if !tools.is_empty() {
        body["tools"] = json!([{
            "functionDeclarations": tools
                .iter()
                .map(|tool| json!({
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }))
                .collect::<Vec<_>>(),
        }]);
    }
    body
}

#[derive(Deserialize)]
struct GeminiReply {
    #[serde(default)]
    candidates: Vec<GeminiCandidate>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: Option<GeminiContent>,
}

#[derive(Deserialize)]
struct GeminiContent {
    #[serde(default)]
    parts: Vec<GeminiPart>,
}

#[derive(Deserialize)]
struct GeminiPart {
    #[serde(default)]
    text: Option<String>,
    /// Set on the reasoning parts of a thinking model.
    #[serde(default)]
    thought: Option<bool>,
    #[serde(rename = "functionCall", default)]
    function_call: Option<GeminiCall>,
}

#[derive(Deserialize)]
struct GeminiCall {
    name: String,
    #[serde(default)]
    args: Option<Value>,
}

#[derive(Default)]
struct GeminiStream {
    text: String,
    calls: Vec<ToolCall>,
}

impl GeminiStream {
    fn apply(&mut self, payload: &str, on_delta: &mut dyn FnMut(Delta, &str)) -> Result<()> {
        let reply: GeminiReply =
            serde_json::from_str(payload).context("unexpected gemini stream event")?;
        for part in reply
            .candidates
            .into_iter()
            .flat_map(|candidate| candidate.content)
            .flat_map(|content| content.parts)
        {
            if let Some(text) = part.text {
                if part.thought.unwrap_or(false) {
                    on_delta(Delta::Thinking, &text);
                } else {
                    self.text.push_str(&text);
                    on_delta(Delta::Text, &text);
                }
            }
            // Gemini sends each call whole, with no id of its own.
            if let Some(call) = part.function_call {
                self.calls.push(ToolCall {
                    id: format!("call_{}", self.calls.len()),
                    name: call.name,
                    arguments: call.args.unwrap_or_else(|| json!({})),
                });
            }
        }
        Ok(())
    }

    fn finish(self) -> Response {
        Response {
            text: (!self.text.is_empty()).then_some(self.text),
            tool_calls: self.calls,
        }
    }
}

async fn stream_gemini(
    response: &mut reqwest::Response,
    on_delta: &mut dyn FnMut(Delta, &str),
) -> Result<Response> {
    let mut stream = GeminiStream::default();
    read_sse(response, |payload| stream.apply(payload, on_delta)).await?;
    Ok(stream.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records the deltas a stream emits, in order.
    #[derive(Default)]
    struct Sink(Vec<(Delta, String)>);

    impl Sink {
        fn callback(&mut self) -> impl FnMut(Delta, &str) + '_ {
            |delta, chunk| self.0.push((delta, chunk.to_string()))
        }

        fn text(&self) -> String {
            self.0
                .iter()
                .filter(|(delta, _)| *delta == Delta::Text)
                .map(|(_, chunk)| chunk.as_str())
                .collect()
        }

        fn thinking(&self) -> usize {
            self.0
                .iter()
                .filter(|(delta, _)| *delta == Delta::Thinking)
                .count()
        }
    }

    #[test]
    fn openai_streams_text_and_reasoning() {
        let mut stream = OpenAiStream::default();
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
        assert_eq!(sink.text(), "Hello");
        assert_eq!(sink.thinking(), 1);
        assert!(response.tool_calls.is_empty());
    }

    #[test]
    fn openai_assembles_fragmented_tool_calls() {
        let mut stream = OpenAiStream::default();
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
    fn openai_keeps_unparseable_arguments_empty() {
        let mut stream = OpenAiStream::default();
        let mut sink = Sink::default();
        let mut callback = sink.callback();
        stream
            .apply(
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"read","arguments":"not json"}}]}}]}"#,
                &mut callback,
            )
            .unwrap();

        assert_eq!(stream.finish().tool_calls[0].arguments, json!({}));
    }

    #[test]
    fn anthropic_streams_text_tool_input_and_thinking() {
        let mut stream = AnthropicStream::default();
        let mut sink = Sink::default();
        let mut callback = sink.callback();
        for payload in [
            r#"{"type":"message_start","message":{"id":"m"}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"let me see"}}"#,
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Look"}}"#,
            r#"{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"edit"}}"#,
            r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}"#,
            r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"a.rs\"}"}}"#,
            r#"{"type":"message_stop"}"#,
        ] {
            stream.apply(payload, &mut callback).unwrap();
        }
        drop(callback);
        let response = stream.finish();

        assert_eq!(response.text.as_deref(), Some("Look"));
        assert_eq!(sink.text(), "Look");
        assert_eq!(sink.thinking(), 1);
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].id, "toolu_1");
        assert_eq!(response.tool_calls[0].name, "edit");
        assert_eq!(response.tool_calls[0].arguments, json!({"path": "a.rs"}));
    }

    #[test]
    fn gemini_streams_text_and_calls() {
        let mut stream = GeminiStream::default();
        let mut sink = Sink::default();
        let mut callback = sink.callback();
        for payload in [
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"thinking","thought":true}]}}]}"#,
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":"Hi"}]}}]}"#,
            r#"{"candidates":[{"content":{"role":"model","parts":[{"text":" there"}]}}]}"#,
            r#"{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"read","args":{"path":"a.rs"}}}]}}]}"#,
        ] {
            stream.apply(payload, &mut callback).unwrap();
        }
        drop(callback);
        let response = stream.finish();

        assert_eq!(response.text.as_deref(), Some("Hi there"));
        assert_eq!(sink.text(), "Hi there");
        assert_eq!(sink.thinking(), 1);
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].name, "read");
        assert_eq!(response.tool_calls[0].arguments, json!({"path": "a.rs"}));
    }
}
