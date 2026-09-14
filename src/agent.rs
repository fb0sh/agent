//! The agent loop: call the model, run the tools it asks for, repeat.

use std::io::{self, IsTerminal, Write};

use anyhow::{Result, bail};

use crate::llm::{Delta, Llm, Message, Response};
use crate::tools::{ToolDefinition, Tools, clip};

const SYSTEM_PROMPT: &str = "\
You are a coding agent working in the current directory.

Use tools to inspect and modify files and execute commands when needed.

Available tools:
- read
- write
- edit
- exec

Inspect relevant files before editing.
Prefer edit for small changes and write for complete files.
Run relevant checks after modifications.";

pub struct Agent {
    llm: Llm,
    tools: Tools,
    definitions: Vec<ToolDefinition>,
    messages: Vec<Message>,
    max_iterations: usize,
    /// Show full tool calls, their output and the model's reasoning.
    verbose: bool,
}

impl Agent {
    pub fn new(llm: Llm, tools: Tools, max_iterations: usize, verbose: bool) -> Self {
        let definitions = tools.definitions();
        Self {
            llm,
            tools,
            definitions,
            messages: vec![Message::System(SYSTEM_PROMPT.to_string())],
            max_iterations,
            verbose,
        }
    }

    /// Run one task to completion and return the model's final text.
    ///
    /// Answer text and reasoning are passed to `on_delta` as they arrive, so the
    /// caller decides where they go; the return value is the same final text.
    pub async fn run(
        &mut self,
        task: &str,
        on_delta: &mut dyn FnMut(Delta, &str),
    ) -> Result<String> {
        self.messages.push(Message::User(task.to_string()));

        // Set while a streamed chunk has left a line half-written, so our own
        // trace output starts on a fresh line instead of gluing onto it.
        let mut line_open = false;
        // Reasoning is only displayed on a terminal, and only when asked for.
        let show_thinking = self.verbose && on_terminal();

        for _ in 0..self.max_iterations {
            // A model turn can take a while; say so instead of showing a blank screen.
            break_line(&mut line_open);
            status(&format!("waiting for {} …", self.llm.model));
            let mut started = false;
            let response = self
                .llm
                .chat(&self.messages, &self.definitions, &mut |delta, chunk| {
                    if !started {
                        started = true;
                        clear_status();
                    }
                    line_open = !chunk.ends_with('\n') && (delta == Delta::Text || show_thinking);
                    on_delta(delta, chunk);
                })
                .await;
            if !started {
                clear_status();
            }
            let Response { text, tool_calls } = response?;

            // No tool calls means the model is done talking.
            if tool_calls.is_empty() {
                return Ok(text.unwrap_or_default());
            }

            // Tools run one at a time on purpose: write/edit/exec depend on each other.
            let mut results = Vec::with_capacity(tool_calls.len());
            for call in &tool_calls {
                break_line(&mut line_open);
                if self.verbose {
                    eprintln!(
                        "→ {}({})",
                        call.name,
                        clip(&call.arguments.to_string(), 160)
                    );
                } else {
                    // One short line per step: enough to see progress, no wall of JSON.
                    eprintln!("→ {}", call.name);
                }
                let result = self.tools.execute(call).await;
                if self.verbose {
                    break_line(&mut line_open);
                    // At most a few lines of output, so the screen never scrolls away.
                    for line in result.output.lines().take(TRACE_LINES) {
                        eprintln!("  {}", clip(line, 160));
                    }
                }
                results.push(result);
            }

            self.messages.push(Message::Assistant {
                text: text.unwrap_or_default(),
                calls: tool_calls,
            });
            self.messages.push(Message::Tools(results));
        }

        bail!(
            "agent stopped after {} steps without a final answer",
            self.max_iterations
        )
    }
}

/// How much tool output a verbose trace shows, so the screen never scrolls away.
const TRACE_LINES: usize = 3;

/// True when progress output is worth showing, i.e. stderr is a terminal.
fn on_terminal() -> bool {
    io::stderr().is_terminal()
}

/// End a line that a streamed chunk left open, so later output is not glued to it.
fn break_line(line_open: &mut bool) {
    if *line_open {
        eprintln!();
        *line_open = false;
    }
}

/// A transient line on stderr, so pipes stay clean.
fn status(text: &str) {
    if on_terminal() {
        eprint!("\r\x1b[2K{text}");
        let _ = io::stderr().flush();
    }
}

fn clear_status() {
    if on_terminal() {
        eprint!("\r\x1b[2K");
        let _ = io::stderr().flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Duration;

    use serde_json::{Value, json};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    use crate::llm::Api;

    /// Wrap OpenAI stream events as an SSE body.
    fn sse(events: &[String]) -> String {
        events
            .iter()
            .map(|event| format!("data: {event}\n\n"))
            .collect()
    }

    fn text_reply(text: &str) -> String {
        sse(&[json!({"choices": [{"delta": {"content": text}}]}).to_string()])
    }

    fn tool_reply(id: &str, name: &str, arguments: &Value) -> String {
        sse(&[json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": id, "function": {"name": name, "arguments": arguments.to_string()}}
        ]}}]})
        .to_string()])
    }

    /// A throwaway OpenAI-compatible server that answers with `replies` in order.
    async fn serve(replies: Vec<String>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            for reply in replies {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                read_request(&mut socket).await;
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{reply}",
                    reply.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.shutdown().await;
                // Hold the socket until the client is done, then accept the next one.
                let mut sink = [0u8; 1024];
                let _ = tokio::time::timeout(Duration::from_secs(5), socket.read(&mut sink)).await;
            }
        });
        format!("http://{address}")
    }

    async fn read_request(socket: &mut TcpStream) {
        let mut request = Vec::new();
        let mut chunk = [0u8; 4096];
        loop {
            let Ok(read) = socket.read(&mut chunk).await else {
                return;
            };
            if read == 0 {
                return;
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
                return;
            }
        }
    }

    fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack.windows(needle.len()).position(|w| w == needle)
    }

    async fn agent(replies: Vec<String>, steps: usize, cwd: PathBuf) -> Agent {
        let base_url = serve(replies).await;
        let llm = Llm::new(
            Api::OpenAi,
            Some("test".into()),
            Some(base_url),
            "test".into(),
        )
        .unwrap();
        Agent::new(llm, Tools::new(cwd), steps, false)
    }

    /// A callback that keeps whatever the agent streams, for assertions.
    #[derive(Default)]
    struct Seen(Vec<(Delta, String)>);

    impl Seen {
        fn callback(&mut self) -> impl FnMut(Delta, &str) + '_ {
            |delta, chunk| self.0.push((delta, chunk.to_string()))
        }
    }

    #[tokio::test]
    async fn returns_a_text_response() {
        let dir = crate::tools::temp_dir("agent-text");
        let mut agent = agent(vec![text_reply("all done")], 4, dir).await;

        assert_eq!(
            agent.run("say hi", &mut |_, _| {}).await.unwrap(),
            "all done"
        );
    }

    #[tokio::test]
    async fn streams_text_as_it_arrives() {
        let dir = crate::tools::temp_dir("agent-stream");
        let replies = sse(&[
            json!({"choices": [{"delta": {"content": "all "}}]}).to_string(),
            json!({"choices": [{"delta": {"content": "done"}}]}).to_string(),
        ]);
        let mut agent = agent(vec![replies], 4, dir).await;

        let mut seen = Seen::default();
        let answer = agent.run("say hi", &mut seen.callback()).await.unwrap();

        assert_eq!(answer, "all done");
        assert_eq!(
            seen.0,
            vec![
                (Delta::Text, "all ".to_string()),
                (Delta::Text, "done".to_string())
            ]
        );
    }

    #[tokio::test]
    async fn streams_reasoning_separately_from_the_answer() {
        let dir = crate::tools::temp_dir("agent-thinking");
        let replies = sse(&[
            json!({"choices": [{"delta": {"reasoning": "thinking"}}]}).to_string(),
            json!({"choices": [{"delta": {"content": "answer"}}]}).to_string(),
        ]);
        let mut agent = agent(vec![replies], 4, dir).await;

        let mut seen = Seen::default();
        let answer = agent.run("think", &mut seen.callback()).await.unwrap();

        assert_eq!(answer, "answer");
        assert_eq!(
            seen.0,
            vec![
                (Delta::Thinking, "thinking".to_string()),
                (Delta::Text, "answer".to_string())
            ]
        );
    }

    #[tokio::test]
    async fn reassembles_tool_arguments_split_across_events() {
        let dir = crate::tools::temp_dir("agent-split-args");
        let mut agent = agent(
            vec![
                sse(&[
                    json!({"choices": [{"delta": {"tool_calls": [
                        {"index": 0, "id": "call_1", "function": {"name": "write", "arguments": "{\"path\":\"a.tx"}}
                    ]}}]})
                    .to_string(),
                    json!({"choices": [{"delta": {"tool_calls": [
                        {"index": 0, "function": {"arguments": "t\",\"content\":\"hi\"}"}}
                    ]}}]})
                    .to_string(),
                ]),
                text_reply("wrote it"),
            ],
            4,
            dir.clone(),
        )
        .await;

        assert_eq!(
            agent.run("write", &mut |_, _| {}).await.unwrap(),
            "wrote it"
        );
        assert_eq!(
            tokio::fs::read_to_string(dir.join("a.txt")).await.unwrap(),
            "hi"
        );
    }

    #[tokio::test]
    async fn executes_one_tool_call() {
        let dir = crate::tools::temp_dir("agent-one-tool");
        let mut agent = agent(
            vec![
                tool_reply(
                    "call_1",
                    "write",
                    &json!({"path": "a.txt", "content": "hi"}),
                ),
                text_reply("wrote it"),
            ],
            4,
            dir.clone(),
        )
        .await;

        assert_eq!(
            agent.run("create a.txt", &mut |_, _| {}).await.unwrap(),
            "wrote it"
        );
        assert_eq!(
            tokio::fs::read_to_string(dir.join("a.txt")).await.unwrap(),
            "hi"
        );
    }

    #[tokio::test]
    async fn loops_over_several_rounds_of_tool_calls() {
        let dir = crate::tools::temp_dir("agent-multi-round");
        tokio::fs::write(dir.join("a.txt"), "before\n")
            .await
            .unwrap();
        let mut agent = agent(
            vec![
                tool_reply("call_1", "read", &json!({"path": "a.txt"})),
                tool_reply(
                    "call_2",
                    "edit",
                    &json!({"path": "a.txt", "old": "before", "new": "after"}),
                ),
                tool_reply("call_3", "exec", &json!({"command": "cat a.txt"})),
                text_reply("done"),
            ],
            6,
            dir.clone(),
        )
        .await;

        assert_eq!(
            agent.run("rename it", &mut |_, _| {}).await.unwrap(),
            "done"
        );
        assert_eq!(
            tokio::fs::read_to_string(dir.join("a.txt")).await.unwrap(),
            "after\n"
        );
    }

    #[tokio::test]
    async fn feeds_tool_errors_back_to_the_model() {
        let dir = crate::tools::temp_dir("agent-tool-error");
        let mut agent = agent(
            vec![
                tool_reply("call_1", "read", &json!({"path": "missing.txt"})),
                tool_reply("call_2", "read", &json!({"path": "nope"})),
                text_reply("gave up"),
            ],
            4,
            dir,
        )
        .await;

        assert_eq!(
            agent.run("read it", &mut |_, _| {}).await.unwrap(),
            "gave up"
        );
    }

    #[tokio::test]
    async fn stops_at_the_maximum_number_of_steps() {
        let dir = crate::tools::temp_dir("agent-max-steps");
        let mut agent = agent(
            vec![
                tool_reply("call_1", "read", &json!({"path": "a.txt"})),
                tool_reply("call_2", "read", &json!({"path": "a.txt"})),
            ],
            2,
            dir,
        )
        .await;

        let error = agent.run("loop forever", &mut |_, _| {}).await.unwrap_err();
        assert!(error.to_string().contains("2 steps"), "{error}");
    }
}
