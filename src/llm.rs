//! The only place that knows about API protocols.
//!
//! Everything above this module speaks [`Message`] / [`Response`]. The JSON
//! shaping for each protocol lives in one `*_body` + `parse_*` pair below.
//! There is no provider registry: a protocol is a variant, a call is a `match`.

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

    pub async fn chat(&self, messages: &[Message], tools: &[ToolDefinition]) -> Result<Response> {
        let response = match self.api {
            Api::OpenAi => self.openai(messages, tools).await?,
            Api::Anthropic => self.anthropic(messages, tools).await?,
            Api::Gemini => self.gemini(messages, tools).await?,
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
            Api::Gemini => format!("{base}/v1beta/models/{}:generateContent", self.model),
        }
    }

    /// One POST per turn, with the auth scheme of the protocol, and API errors
    /// surfaced with their response body instead of a bare status code.
    async fn post(&self, body: &Value) -> Result<Value> {
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
        let text = response
            .text()
            .await
            .context("failed to read the response body")?;
        if !status.is_success() {
            bail!(
                "{} error {status}: {}",
                self.api,
                clip(&text, ERROR_BODY_LIMIT)
            );
        }
        serde_json::from_str(&text)
            .with_context(|| format!("{} returned invalid JSON: {}", self.api, clip(&text, 500)))
    }

    // ---------------------------------------------------------------- openai

    async fn openai(&self, messages: &[Message], tools: &[ToolDefinition]) -> Result<Response> {
        let body = self
            .post(&openai_body(&self.model, messages, tools))
            .await?;
        parse_openai(body)
    }

    // ------------------------------------------------------------- anthropic

    async fn anthropic(&self, messages: &[Message], tools: &[ToolDefinition]) -> Result<Response> {
        let body = self
            .post(&anthropic_body(&self.model, messages, tools))
            .await?;
        parse_anthropic(body)
    }

    // ---------------------------------------------------------------- gemini

    async fn gemini(&self, messages: &[Message], tools: &[ToolDefinition]) -> Result<Response> {
        let body = self.post(&gemini_body(messages, tools)).await?;
        parse_gemini(body)
    }
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

    let mut body = json!({"model": model, "messages": msgs});
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
struct OpenAiReply {
    choices: Vec<OpenAiChoice>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
}

#[derive(Deserialize)]
struct OpenAiMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiCall>>,
}

#[derive(Deserialize)]
struct OpenAiCall {
    #[serde(default)]
    id: String,
    function: OpenAiFunction,
}

#[derive(Deserialize)]
struct OpenAiFunction {
    name: String,
    /// Usually a JSON string, but some gateways send an object.
    arguments: Option<Value>,
}

fn parse_openai(body: Value) -> Result<Response> {
    let reply: OpenAiReply =
        serde_json::from_value(body).context("unexpected openai response shape")?;
    let Some(choice) = reply.choices.into_iter().next() else {
        return Ok(Response::default());
    };
    let tool_calls = choice
        .message
        .tool_calls
        .unwrap_or_default()
        .into_iter()
        .map(|call| {
            let arguments = parse_arguments(call.function.arguments.as_ref());
            ToolCall {
                id: call.id,
                name: call.function.name,
                arguments,
            }
        })
        .collect();
    Ok(Response {
        text: choice.message.content.filter(|text| !text.is_empty()),
        tool_calls,
    })
}

fn parse_arguments(arguments: Option<&Value>) -> Value {
    match arguments {
        Some(Value::String(text)) if !text.trim().is_empty() => {
            serde_json::from_str(text).unwrap_or_else(|_| json!({}))
        }
        Some(Value::Object(_)) => arguments.cloned().unwrap_or_else(|| json!({})),
        _ => json!({}),
    }
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
struct AnthropicReply {
    content: Vec<AnthropicBlock>,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
    },
    #[serde(other)]
    Other,
}

fn parse_anthropic(body: Value) -> Result<Response> {
    let reply: AnthropicReply =
        serde_json::from_value(body).context("unexpected anthropic response shape")?;
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for block in reply.content {
        match block {
            AnthropicBlock::Text { text: chunk } => text.push_str(&chunk),
            AnthropicBlock::ToolUse { id, name, input } => tool_calls.push(ToolCall {
                id,
                name,
                arguments: input,
            }),
            AnthropicBlock::Other => {}
        }
    }
    Ok(Response {
        text: (!text.is_empty()).then_some(text),
        tool_calls,
    })
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
    #[serde(rename = "functionCall", default)]
    function_call: Option<GeminiCall>,
}

#[derive(Deserialize)]
struct GeminiCall {
    name: String,
    #[serde(default)]
    args: Option<Value>,
}

fn parse_gemini(body: Value) -> Result<Response> {
    let reply: GeminiReply =
        serde_json::from_value(body).context("unexpected gemini response shape")?;
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for part in reply
        .candidates
        .into_iter()
        .flat_map(|candidate| candidate.content)
        .flat_map(|content| content.parts)
    {
        if let Some(chunk) = part.text {
            text.push_str(&chunk);
        }
        if let Some(call) = part.function_call {
            tool_calls.push(ToolCall {
                id: format!("call_{}", tool_calls.len()),
                name: call.name,
                arguments: call.args.unwrap_or_else(|| json!({})),
            });
        }
    }
    Ok(Response {
        text: (!text.is_empty()).then_some(text),
        tool_calls,
    })
}
