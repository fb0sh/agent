//! The agent loop: call the model, run the tools it asks for, repeat.

use std::io::{self, IsTerminal, Write};

use anyhow::{Result, bail};

use crate::llm::{Llm, Message, Response};
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
}

impl Agent {
    pub fn new(llm: Llm, tools: Tools, max_iterations: usize) -> Self {
        let definitions = tools.definitions();
        Self {
            llm,
            tools,
            definitions,
            messages: vec![Message::System(SYSTEM_PROMPT.to_string())],
            max_iterations,
        }
    }

    /// Run one task to completion and return the model's final text.
    pub async fn run(&mut self, task: &str) -> Result<String> {
        self.messages.push(Message::User(task.to_string()));

        for _ in 0..self.max_iterations {
            // A model turn can take a while; say so instead of showing a blank screen.
            status(&format!("waiting for {} …", self.llm.model));
            let response = self.llm.chat(&self.messages, &self.definitions).await;
            clear_status();
            let Response { text, tool_calls } = response?;

            // No tool calls means the model is done talking.
            if tool_calls.is_empty() {
                return Ok(text.unwrap_or_default());
            }
            if let Some(thought) = text.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
                eprintln!("{thought}");
            }

            // Tools run one at a time on purpose: write/edit/exec depend on each other.
            let mut results = Vec::with_capacity(tool_calls.len());
            for call in &tool_calls {
                eprintln!(
                    "→ {}({})",
                    call.name,
                    clip(&call.arguments.to_string(), 160)
                );
                let result = self.tools.execute(call).await;
                eprintln!("  {}", first_line(&result.output));
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

fn first_line(text: &str) -> &str {
    text.lines().next().unwrap_or("(no output)")
}

/// A transient line on stderr, shown only on a terminal so pipes stay clean.
fn status(text: &str) {
    if io::stderr().is_terminal() {
        eprint!("\r\x1b[2K{text}");
        let _ = io::stderr().flush();
    }
}

fn clear_status() {
    if io::stderr().is_terminal() {
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

    fn text_reply(text: &str) -> String {
        json!({"choices": [{"message": {"role": "assistant", "content": text}}]}).to_string()
    }

    fn tool_reply(id: &str, name: &str, arguments: &Value) -> String {
        json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments.to_string()},
                    }]
                }
            }]
        })
        .to_string()
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
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{reply}",
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
        Agent::new(llm, Tools::new(cwd), steps)
    }

    #[tokio::test]
    async fn returns_a_text_response() {
        let dir = crate::tools::temp_dir("agent-text");
        let mut agent = agent(vec![text_reply("all done")], 4, dir).await;

        assert_eq!(agent.run("say hi").await.unwrap(), "all done");
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

        assert_eq!(agent.run("create a.txt").await.unwrap(), "wrote it");
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

        assert_eq!(agent.run("rename it").await.unwrap(), "done");
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

        assert_eq!(agent.run("read it").await.unwrap(), "gave up");
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

        let error = agent.run("loop forever").await.unwrap_err();
        assert!(error.to_string().contains("2 steps"), "{error}");
    }
}
