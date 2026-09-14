//! `mini-agent "fix the build"` — a thin CLI over the `agent` library.

use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;

use agent::{Agent, Delta, Llm, Output, Tools, llm};
use anyhow::{Context, Result, bail};
use clap::Parser;

/// A minimal coding agent that works in the current directory with four tools:
/// read, write, edit and exec.
#[derive(Debug, Parser)]
#[command(name = "mini-agent", version, about, max_term_width = 100)]
struct Args {
    /// What the agent should do.
    #[arg(required = true, value_name = "TASK")]
    task: Vec<String>,

    /// Model name.
    #[arg(long, env = "MINI_AGENT_MODEL", value_name = "MODEL")]
    model: Option<String>,

    /// API base URL of any OpenAI-compatible endpoint.
    #[arg(
        long,
        env = "MINI_AGENT_BASE_URL",
        value_name = "URL",
        hide_env_values = true
    )]
    base_url: Option<String>,

    /// API key (default: OPENAI_API_KEY).
    #[arg(
        long,
        env = "MINI_AGENT_API_KEY",
        value_name = "KEY",
        hide_env_values = true
    )]
    api_key: Option<String>,

    /// Directory all tools work in.
    #[arg(short = 'C', long, env = "MINI_AGENT_CWD", value_name = "DIR")]
    cwd: Option<PathBuf>,

    /// Give up after this many model turns.
    #[arg(
        long,
        env = "MINI_AGENT_MAX_ITERATIONS",
        default_value_t = 32,
        value_name = "N"
    )]
    max_iterations: usize,

    /// Show each tool call, the tool output and the model's reasoning.
    #[arg(short, long, env = "MINI_AGENT_VERBOSE")]
    verbose: bool,
}

#[tokio::main]
async fn main() {
    // A `.env` in the working directory fills in variables that are not already set.
    let _ = dotenvy::dotenv();

    if let Err(error) = run().await {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let args = Args::parse();

    let task = args.task.join(" ");
    if task.trim().is_empty() {
        bail!("no task given");
    }

    let cwd = match &args.cwd {
        Some(dir) => dir
            .canonicalize()
            .with_context(|| format!("invalid --cwd {}", dir.display()))?,
        None => std::env::current_dir().context("cannot determine the current directory")?,
    };

    let api_key = args
        .api_key
        .filter(|key| !key.trim().is_empty())
        .or_else(|| {
            std::env::var(llm::API_KEY_ENV)
                .ok()
                .filter(|key| !key.trim().is_empty())
        })
        .with_context(|| format!("no API key: pass --api-key or set {}", llm::API_KEY_ENV))?;

    let llm = Llm::new(args.model, args.base_url, api_key)?;
    let tools = Tools::new(cwd);
    let output = if args.verbose {
        Output::Verbose
    } else {
        Output::Progress
    };
    let mut agent = Agent::new(llm, tools, args.max_iterations, output);

    // The answer streams straight to stdout. Reasoning is only worth showing on a
    // terminal, and only when the user asked for detail.
    let show_thinking = args.verbose;
    let mut streamed = false;
    let mut ended_with_newline = true;
    let mut thinking_open = false;
    let answer = agent
        .run(&task, &mut |delta, chunk| {
            match delta {
                Delta::Text => {
                    // Reasoning came first: end its dim line before the answer.
                    if thinking_open {
                        eprintln!();
                        thinking_open = false;
                    }
                    streamed = true;
                    ended_with_newline = chunk.ends_with('\n');
                    let mut out = io::stdout().lock();
                    let _ = out.write_all(chunk.as_bytes());
                    let _ = out.flush();
                }
                Delta::Thinking if show_thinking && io::stderr().is_terminal() => {
                    let mut err = io::stderr().lock();
                    let _ = write!(err, "\x1b[2m{chunk}\x1b[0m");
                    let _ = err.flush();
                    thinking_open = !chunk.ends_with('\n');
                }
                Delta::Thinking => {}
            }
        })
        .await?;
    if thinking_open {
        eprintln!();
    }

    if streamed {
        if !ended_with_newline {
            println!();
        }
    } else if answer.trim().is_empty() {
        eprintln!("(the model finished without an answer)");
    } else {
        // Nothing was streamed, so print the answer we got.
        println!("{}", answer.trim_end());
    }
    Ok(())
}
