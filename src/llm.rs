//! The OpenAI-compatible protocol: request shaping, streaming, error handling.
//!
//! Everything above this module speaks [`Message`] / [`Response`]. Any service
//! that speaks OpenAI Chat Completions works — OpenAI, DeepSeek, OpenRouter,
//! Qwen, GLM, Moonshot, Groq, Together, Fireworks, vLLM, Ollama, LM Studio,
//! proxies — by pointing [`Llm::base_url`] at it. No vendor is special-cased.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::tools::{ToolDefinition, clip};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
const ERROR_BODY_LIMIT: usize = 2000;

pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
pub const DEFAULT_MODEL: &str = "gpt-4o-mini";
/// Conventional environment variable holding the key.
pub const API_KEY_ENV: &str = "OPENAI_API_KEY";

/// One entry of the conversation, in the shape the protocol needs.
///
/// Text and tool calls share one assistant turn, and parallel tool results share
/// one turn, so each variant maps to one or more messages without regrouping.
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
    pub model: String,
    pub base_url: String,
    pub api_key: String,
    client: reqwest::Client,
}

impl Llm {
    /// `model` and `base_url` fall back to OpenAI's own values.
    pub fn new(model: Option<String>, base_url: Option<String>, api_key: String) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .context("failed to build the HTTP client")?;
        Ok(Self {
            model: model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            base_url: base_url
                .unwrap_or_else(|| DEFAULT_BASE_URL.to_string())
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
        let mut response = self.send(&body(&self.model, messages, tools)).await?;

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
        read_sse(&mut response, |payload| stream.apply(payload, on_delta)).await?;
        let response = stream.finish();
        if response.text.is_none() && response.tool_calls.is_empty() {
            bail!("the model returned neither text nor tool calls");
        }
        Ok(response)
    }

    /// One POST per turn; API errors carry their response body, which is where
    /// these services explain what went wrong.
    async fn send(&self, body: &Value) -> Result<reqwest::Response> {
        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(body)
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

fn body(model: &str, messages: &[Message], tools: &[ToolDefinition]) -> Value {
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
                serde_json::from_str(&self.arguments).unwrap_or_else(|_| json!({}))
            },
        }
    }
}

/// The state of one streamed turn.
#[derive(Default)]
struct Stream {
    text: String,
    calls: Vec<CallBuilder>,
}

impl Stream {
    fn apply(&mut self, payload: &str, on_delta: &mut dyn FnMut(Delta, &str)) -> Result<()> {
        let chunk: Chunk = serde_json::from_str(payload).context("unexpected stream event")?;
        for choice in chunk.choices {
            if let Some(text) = choice.delta.content.filter(|text| !text.is_empty()) {
                self.text.push_str(&text);
                on_delta(Delta::Text, &text);
            }
            if let Some(thinking) = choice.delta.reasoning.filter(|text| !text.is_empty()) {
                on_delta(Delta::Thinking, &thinking);
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
        Response {
            text: (!self.text.is_empty()).then_some(self.text),
            tool_calls: self.calls.into_iter().map(CallBuilder::finish).collect(),
        }
    }
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
        assert_eq!(sink.text(), "Hello");
        assert_eq!(sink.thinking(), 1);
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
    fn keeps_unparseable_arguments_empty() {
        let mut stream = Stream::default();
        let mut sink = Sink::default();
        let mut callback = sink.callback();
        stream
            .apply(
                r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c","function":{"name":"read","arguments":"not json"}}]}}]}"#,
                &mut callback,
            )
            .unwrap();
        drop(callback);

        assert_eq!(stream.finish().tool_calls[0].arguments, json!({}));
    }

    #[test]
    fn builds_a_request_with_tools_and_history() {
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
            },
            Message::Tools(vec![ToolResult {
                id: "call_1".into(),
                name: "read".into(),
                output: "fn main() {}".into(),
            }]),
        ];
        let tools = vec![ToolDefinition {
            name: "read",
            description: "read a file",
            parameters: json!({"type": "object"}),
        }];
        let body = body("test-model", &messages, &tools);

        assert_eq!(body["model"], "test-model");
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
    }
}
