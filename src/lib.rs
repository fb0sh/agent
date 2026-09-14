//! A minimal coding agent: a model loop over four tools (`read`, `write`,
//! `edit`, `exec`), speaking the OpenAI-compatible Chat Completions API — so it
//! works with OpenAI, DeepSeek, OpenRouter, Qwen, Groq, Ollama, vLLM and any
//! other compatible endpoint.
//!
//! The library stays quiet: it never writes to the terminal on its own. Answer
//! text and reasoning arrive through the callback passed to [`Agent::run`], and
//! progress reporting is opt-in through [`Output`]. The `mini-agent` binary in
//! this package is a thin CLI over exactly this API.
//!
//! ```no_run
//! use agent::{Agent, Llm, Output, Tools};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let llm = Llm::new(Some("deepseek-chat".into()), None, "sk-…".into())?;
//! let tools = Tools::new(std::env::current_dir()?);
//! let mut agent = Agent::new(llm, tools, 32, Output::Quiet);
//!
//! let mut seen = String::new();
//! let answer = agent
//!     .run("fix the build", &mut |_, chunk| seen.push_str(chunk))
//!     .await?;
//! println!("{answer}");
//! # Ok(())
//! # }
//! ```

pub mod agent;
pub mod llm;
pub mod tools;

pub use agent::{Agent, Output};
pub use llm::{Delta, Llm, Message, Response, ToolCall, ToolResult};
pub use tools::{ToolDefinition, Tools};
