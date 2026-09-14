//! A minimal coding agent: a model loop over four tools (`read`, `write`,
//! `edit`, `exec`).
//!
//! This module holds the crate's vocabulary and its two extension points:
//! [`Model`] (the chat protocol) and [`Toolbox`] (the tools). [`Agent`] is
//! generic over both, so a host program can swap either one without touching
//! the loop, and the library itself never writes to the terminal.
//!
//! ```no_run
//! use agent::{Agent, CodingTools, OpenAiCompatible, OpenAiConfig, Event};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let model = OpenAiCompatible::new(
//!     OpenAiCompatible::builder()
//!         .model("deepseek-chat")
//!         .api_key("sk-…")
//!         .param("temperature", 0.2),
//! )?;
//! let tools = CodingTools::new(std::env::current_dir()?);
//! let mut agent = Agent::new(model, tools).max_iterations(32);
//!
//! let answer = agent
//!     .run("fix the build", &mut |event| match event {
//!         Event::Text(chunk) => print!("{chunk}"),
//!         _ => {}
//!     })
//!     .await?;
//! println!("{answer}");
//! # Ok(())
//! # }
//! ```

use std::future::Future;

use anyhow::Result;
use serde_json::Value;

pub mod agent;
pub mod llm;
pub mod tools;

pub use agent::Agent;
pub use llm::{OpenAiCompatible, OpenAiConfig};
pub use tools::CodingTools;

/// One entry of the conversation.
///
/// Text and tool calls share one assistant turn, and the results of parallel
/// tool calls share one turn.
#[derive(Debug, Clone)]
pub enum Message {
    System(String),
    User(String),
    Assistant { text: String, calls: Vec<ToolCall> },
    Tools(Vec<ToolResult>),
}

/// A tool call the model asked for.
///
/// `arguments` is normally a JSON object. If the model sent something that is
/// not valid JSON, the raw text is kept as a JSON string, so the tool can report
/// what actually arrived instead of a misleading "missing argument".
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// What the model is told after a tool ran.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub id: String,
    pub name: String,
    pub output: String,
}

/// A tool, described the way every OpenAI-compatible model expects to see it.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// What one model turn produced.
#[derive(Debug, Clone, Default)]
pub struct Response {
    pub text: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

/// Something that happens while the agent works.
///
/// A [`Model`] implementation only ever emits [`Event::Text`] and
/// [`Event::Thinking`]; the agent adds the two tool events around its own work.
/// Hosts render these however they like — the library prints nothing.
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

/// Cut `text` down to `max` bytes on a char boundary, marking the cut.
pub(crate) fn clip(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n[output truncated]", &text[..end])
}
