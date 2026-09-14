//! `mini-agent "fix the build"` — a CLI over the `agent` library.
//!
//! Argument parsing and all terminal rendering live here. The library only
//! reports events; how they look is this program's business.

use std::io::{self, IsTerminal, Write};
use std::path::PathBuf;

use agent::{Agent, CodingTools, Event, OpenAiCompatible, OpenAiConfig, llm};
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

    /// API key. Omit it for a local endpoint that needs none.
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

    /// Also show each tool call and its output.
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

    let mut config = OpenAiConfig::default();
    if let Some(base_url) = &args.base_url {
        config = config.base_url(base_url.as_str());
    }
    if let Some(model) = &args.model {
        config = config.model(model.as_str());
    }
    match args.api_key.as_deref() {
        Some(key) if !key.trim().is_empty() => config = config.api_key(key),
        // No key given: fall back to the conventional variable, and otherwise
        // send no Authorization header at all, which local servers expect.
        _ => {
            if let Some(key) = std::env::var(llm::API_KEY_ENV)
                .ok()
                .filter(|key| !key.trim().is_empty())
            {
                config = config.api_key(key);
            }
        }
    }

    let model = OpenAiCompatible::new(config)?;
    let tools = CodingTools::new(cwd);
    let mut agent = Agent::new(model, tools).max_iterations(args.max_iterations);

    // The answer streams to stdout; reasoning and traces are drawn on stderr.
    let verbose = args.verbose;
    let mut line = Line::Clean;
    let mut thinking = String::new();
    let mut streamed = false;
    let mut ended_with_newline = true;

    let result = agent
        .run(&task, &mut |event| match event {
            Event::Text(chunk) => {
                // The answer needs the line: drop the reasoning display first.
                if line == Line::Ours {
                    release(&mut line);
                }
                streamed = true;
                ended_with_newline = chunk.ends_with('\n');
                let mut out = io::stdout().lock();
                let _ = out.write_all(chunk.as_bytes());
                let _ = out.flush();
                line = if ended_with_newline {
                    Line::Clean
                } else {
                    Line::Answer
                };
            }
            Event::Thinking(chunk) => {
                thinking.push_str(chunk);
                draw(&mut line, &format!("\x1b[2m{}\x1b[0m", tail(&thinking)));
            }
            Event::ToolCall { name, arguments } if verbose => {
                release(&mut line);
                eprintln!("→ {}({})", name, preview(&arguments.to_string()));
            }
            Event::ToolResult { output, .. } if verbose => {
                for text in output.lines().take(TRACE_LINES) {
                    eprintln!("  {}", preview(text));
                }
            }
            _ => {}
        })
        .await;

    let answer = match result {
        Ok(answer) => answer,
        Err(error) => {
            release(&mut line);
            return Err(error);
        }
    };
    if line == Line::Ours {
        release(&mut line);
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

// ----------------------------------------------------------------- rendering

/// Reasoning is squeezed into one line this many columns wide, never more.
const THINKING_WIDTH: usize = 60;
/// How much tool output a verbose trace shows, so the screen never scrolls away.
const TRACE_LINES: usize = 3;
/// How much of a tool call's arguments or output line to show.
const PREVIEW_WIDTH: usize = 160;

/// What the last line of the terminal currently holds.
#[derive(Clone, Copy, PartialEq)]
enum Line {
    /// Nothing half-written.
    Clean,
    /// Streamed answer text, which is the user's output and must not be erased.
    Answer,
    /// A line we drew ourselves: ours to overwrite.
    Ours,
}

fn on_terminal() -> bool {
    io::stderr().is_terminal()
}

/// Draw over our own line, or start a fresh one.
fn draw(line: &mut Line, text: &str) {
    release(line);
    if on_terminal() {
        eprint!("{text}");
        let _ = io::stderr().flush();
        *line = Line::Ours;
    }
}

/// Get back to a clean line: erase our own progress, or step past answer text.
fn release(line: &mut Line) {
    match *line {
        Line::Ours if on_terminal() => {
            eprint!("\r\x1b[2K");
            let _ = io::stderr().flush();
        }
        Line::Answer => eprintln!(),
        _ => {}
    }
    *line = Line::Clean;
}

/// The tail of `text` that fits in one line, with newlines flattened so it can
/// never wrap: reasoning scrolls in place instead of scrolling the screen.
fn tail(text: &str) -> String {
    let mut width = 0;
    let mut taken = Vec::new();
    for character in text.chars().rev() {
        let character = if character.is_whitespace() {
            ' '
        } else {
            character
        };
        let columns = if character.is_ascii() { 1 } else { 2 };
        if width + columns > THINKING_WIDTH {
            break;
        }
        width += columns;
        taken.push(character);
    }
    taken.iter().rev().collect()
}

/// One short line of a longer string, for a trace.
fn preview(text: &str) -> String {
    text.chars().take(PREVIEW_WIDTH).collect()
}
