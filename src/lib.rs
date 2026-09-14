//! A minimal coding agent: a model loop over four tools (`read`, `write`,
//! `edit`, `exec`).
//!
//! This module holds the crate's vocabulary and its three extension points:
//! [`Model`] (the chat protocol), [`Toolbox`] (the tools) and [`ContextManager`]
//! (what the model gets to see). [`Agent`] is generic over all three, so a host
//! can replace any of them without touching the loop. The library never writes
//! to the terminal: [`Agent::run`] reports [`Event`]s and the host renders them.
//!
//! ```no_run
//! use agent::{Agent, CodingTools, OpenAiCompatible};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let model = OpenAiCompatible::new(
//!     OpenAiCompatible::builder()
//!         .model("deepseek-chat")
//!         .api_key("sk-…")
//!         .param("temperature", 0.2),
//! )?;
//! let tools = CodingTools::new(std::env::current_dir()?);
//! let mut agent = Agent::new(model, tools);
//!
//! let answer = agent
//!     .run("fix the build", &mut |_event| {})
//!     .await?;
//! println!("{answer}");
//! # Ok(())
//! # }
//! ```

use std::future::Future;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub mod agent;
pub mod context;
pub mod llm;
pub mod tools;

pub use agent::{Agent, AgentHandle, QueueMode, StepOutcome};
pub use context::{KeepLast, NoopContext};
pub use llm::{OpenAiCompatible, OpenAiConfig};
pub use tools::CodingTools;

/// One entry of the conversation.
///
/// Text and tool calls share one assistant turn, and the results of parallel
/// tool calls share one turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    System(String),
    User(String),
    Assistant {
        text: String,
        calls: Vec<ToolCall>,
        /// Provider fields that must be echoed back on the next request, such as
        /// `reasoning_content`. Absent for providers that need none.
        #[serde(default, skip_serializing_if = "Map::is_empty")]
        metadata: Map<String, Value>,
    },
    Tools(Vec<ToolResult>),
}

/// A tool call the model asked for.
///
/// `arguments` is normally a JSON object. If the model sent something that is
/// not valid JSON, the raw text is kept as a JSON string, so the tool can report
/// what actually arrived instead of a misleading "missing argument".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// What the model is told after a tool ran.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub id: String,
    pub name: String,
    pub output: String,
}

/// A tool, described the way every OpenAI-compatible model expects to see it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// What one model turn produced.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Response {
    pub text: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    /// See [`Message::Assistant`]: round-trip fields, not display data.
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub metadata: Map<String, Value>,
}

/// Something that happens while the agent works.
///
/// A [`Model`] implementation only emits [`Event::Text`] and
/// [`Event::Thinking`]; the agent adds the tool events around its own work.
/// Hosts render these however they like.
#[derive(Debug)]
pub enum Event<'a> {
    /// Answer text, as it is generated.
    Text(&'a str),
    /// The model's reasoning, as it is generated.
    Thinking(&'a str),
    /// A tool is about to run.
    ToolCall { name: &'a str, arguments: &'a Value },
    /// A tool finished; `output` is what the model will see.
    ToolResult { name: &'a str, output: &'a str },
    /// A tool the model asked for was dropped because a steering message
    /// arrived first. The model is told the call was cancelled.
    ToolSkipped { name: &'a str },
}

/// A chat model: one protocol implementation per type.
///
/// Implement this to add Anthropic, Gemini, a local runtime or a proxy. The
/// agent never sees the wire format.
pub trait Model {
    /// Run one turn, streaming output through `on_event`, and return what the
    /// model produced.
    fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        on_event: &mut dyn FnMut(Event<'_>),
    ) -> impl Future<Output = Result<Response>>;
}

/// A set of tools: `definitions` is what the model is told they are, `execute`
/// runs one call.
///
/// Returning `Err` is normal — the agent turns it into the tool result text so
/// the model can correct itself.
pub trait Toolbox {
    fn definitions(&self) -> Vec<ToolDefinition>;

    fn execute(&self, call: &ToolCall) -> impl Future<Output = Result<String>>;
}

/// Decides what the model is shown for each turn.
///
/// `prepare` is called right before every model call, with the exact messages
/// that are about to be sent, and may rewrite them: trim, summarise, inject.
/// It is async because a future implementation may need a model call to
/// summarise, even though the built-in ones do not.
pub trait ContextManager {
    fn prepare(&mut self, messages: &mut Vec<Message>) -> impl Future<Output = Result<()>>;
}

/// Cut `text` down to `max` bytes, marking the cut.
pub(crate) fn clip(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    format!("{}\n[output truncated]", truncate(text, max))
}

/// Cut `text` to at most `max` bytes, on a char boundary.
pub(crate) fn truncate(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}
