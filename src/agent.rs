//! The agent loop: call the model, run the tools it asks for, repeat.
//!
//! This is mechanism only: it holds no policy about rendering, retries or
//! context management. Everything a host might want to decide is either a
//! callback ([`Agent::run`]) or an accessor ([`Agent::messages`]).

use anyhow::{Result, bail};

use crate::{Event, Message, Model, Response, ToolDefinition, ToolResult, Toolbox};

/// Used unless the host sets its own with [`Agent::system_prompt`].
pub const DEFAULT_SYSTEM_PROMPT: &str = "\
You are a coding agent working in the current directory.
Use the available tools to inspect, modify, and validate the project.
Inspect relevant files before editing.
Run relevant checks after modifications.";

const DEFAULT_MAX_ITERATIONS: usize = 32;

pub struct Agent<M: Model, T: Toolbox> {
    model: M,
    tools: T,
    system_prompt: String,
    max_iterations: usize,
    messages: Vec<Message>,
}

impl<M: Model, T: Toolbox> Agent<M, T> {
    pub fn new(model: M, tools: T) -> Self {
        Self {
            model,
            tools,
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            max_iterations: DEFAULT_MAX_ITERATIONS,
            messages: Vec::new(),
        }
    }

    /// Replace the system prompt. Set it to `""` to send no system message.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Give up after this many model turns.
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// The conversation so far, so a host can save, restore, trim or inspect it.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn messages_mut(&mut self) -> &mut Vec<Message> {
        &mut self.messages
    }

    /// Forget the conversation, including the system prompt.
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Run one task to completion and return the model's final text.
    ///
    /// The system prompt is seeded only when the conversation is empty, so a
    /// restored session keeps whatever it was saved with.
    pub async fn run(&mut self, task: &str, on_event: &mut dyn FnMut(Event<'_>)) -> Result<String> {
        if self.messages.is_empty() && !self.system_prompt.is_empty() {
            self.messages
                .push(Message::System(self.system_prompt.clone()));
        }
        self.messages.push(Message::User(task.to_string()));

        let definitions: Vec<ToolDefinition> = self.tools.definitions();
        for _ in 0..self.max_iterations {
            let Response { text, tool_calls } = self
                .model
                .chat(&self.messages, &definitions, on_event)
                .await?;

            // No tool calls means the model is done talking.
            if tool_calls.is_empty() {
                let text = text.unwrap_or_default();
                self.messages.push(Message::Assistant {
                    text: text.clone(),
                    calls: Vec::new(),
                });
                return Ok(text);
            }

            // Tools run one at a time on purpose: write/edit/exec depend on each
            // other's effects.
            let mut results = Vec::with_capacity(tool_calls.len());
            for call in &tool_calls {
                on_event(Event::ToolCall {
                    name: &call.name,
                    arguments: &call.arguments,
                });
                let output = match self.tools.execute(call).await {
                    Ok(output) => output,
                    // A failed tool is not a failed run: the model gets to read
                    // the error and correct itself.
                    Err(error) => format!("error: {error:#}"),
                };
                on_event(Event::ToolResult {
                    name: &call.name,
                    output: &output,
                });
                results.push(ToolResult {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    output,
                });
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

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::rc::Rc;

    use serde_json::json;

    use super::*;
    use crate::tools::temp_dir;
    use crate::{CodingTools, ToolCall};

    /// A model that plays back a script and records the prompts it was shown.
    struct Scripted {
        replies: RefCell<VecDeque<Response>>,
        seen: Rc<RefCell<Vec<Vec<Message>>>>,
    }

    impl Scripted {
        fn new(replies: Vec<Response>) -> Self {
            Self {
                replies: RefCell::new(replies.into()),
                seen: Rc::new(RefCell::new(Vec::new())),
            }
        }

        /// A handle to the recorded prompts, since the agent owns the model.
        fn seen(&self) -> Rc<RefCell<Vec<Vec<Message>>>> {
            Rc::clone(&self.seen)
        }

        /// The conversation the model was shown on `turn`.
        fn prompt(seen: &Rc<RefCell<Vec<Vec<Message>>>>, turn: usize) -> Vec<Message> {
            seen.borrow()[turn].clone()
        }
    }

    impl Model for Scripted {
        async fn chat(
            &self,
            messages: &[Message],
            _tools: &[ToolDefinition],
            on_event: &mut dyn FnMut(Event<'_>),
        ) -> Result<Response> {
            self.seen.borrow_mut().push(messages.to_vec());
            let reply = self
                .replies
                .borrow_mut()
                .pop_front()
                .expect("the script ran out of replies");
            if let Some(text) = &reply.text {
                on_event(Event::Text(text));
            }
            Ok(reply)
        }
    }

    fn text(text: &str) -> Response {
        Response {
            text: Some(text.into()),
            tool_calls: Vec::new(),
        }
    }

    fn call(id: &str, name: &str, arguments: serde_json::Value) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }

    fn calls(list: Vec<ToolCall>) -> Response {
        Response {
            text: None,
            tool_calls: list,
        }
    }

    /// Events the agent reported, as compact strings.
    #[derive(Default)]
    struct Seen(Vec<String>);

    impl Seen {
        fn callback(&mut self) -> impl FnMut(Event<'_>) + '_ {
            |event| {
                self.0.push(match event {
                    Event::Text(text) => format!("text:{text}"),
                    Event::Thinking(text) => format!("thinking:{text}"),
                    Event::ToolCall { name, .. } => format!("call:{name}"),
                    Event::ToolResult { output, .. } => format!("result:{output}"),
                })
            }
        }
    }

    #[tokio::test]
    async fn returns_the_final_text_and_keeps_history() {
        let dir = temp_dir("agent-text");
        let model = Scripted::new(vec![text("all done")]);
        let mut agent = Agent::new(model, CodingTools::new(dir));

        let mut seen = Seen::default();
        let answer = agent.run("say hi", &mut seen.callback()).await.unwrap();

        assert_eq!(answer, "all done");
        assert_eq!(seen.0, ["text:all done"]);
        let roles: Vec<&str> = agent
            .messages()
            .iter()
            .map(|message| match message {
                Message::System(_) => "system",
                Message::User(_) => "user",
                Message::Assistant { .. } => "assistant",
                Message::Tools(_) => "tools",
            })
            .collect();
        assert_eq!(roles, ["system", "user", "assistant"]);
    }

    #[tokio::test]
    async fn runs_a_tool_and_feeds_the_result_back() {
        let dir = temp_dir("agent-one-tool");
        let model = Scripted::new(vec![
            calls(vec![call(
                "call_1",
                "write",
                json!({"path": "a.txt", "content": "hi"}),
            )]),
            text("wrote it"),
        ]);
        let seen_prompts = model.seen();
        let mut agent = Agent::new(model, CodingTools::new(dir.clone()));

        let mut seen = Seen::default();
        let answer = agent
            .run("create a.txt", &mut seen.callback())
            .await
            .unwrap();

        assert_eq!(answer, "wrote it");
        assert_eq!(
            tokio::fs::read_to_string(dir.join("a.txt")).await.unwrap(),
            "hi"
        );
        assert!(seen.0[0].starts_with("call:write"), "{:?}", seen.0);
        assert!(seen.0[1].contains("wrote 2 bytes"), "{:?}", seen.0);

        // The second turn is shown the call and its result.
        let prompt = Scripted::prompt(&seen_prompts, 1);
        assert!(matches!(prompt[2], Message::Assistant { ref calls, .. } if calls.len() == 1));
        assert!(matches!(prompt[3], Message::Tools(ref results) if results[0].id == "call_1"));
    }

    #[tokio::test]
    async fn runs_several_rounds_and_several_calls() {
        let dir = temp_dir("agent-rounds");
        tokio::fs::write(dir.join("a.txt"), "before\n")
            .await
            .unwrap();
        let model = Scripted::new(vec![
            calls(vec![
                call("call_1", "read", json!({"path": "a.txt"})),
                call("call_2", "exec", json!({"command": "true"})),
            ]),
            calls(vec![call(
                "call_3",
                "edit",
                json!({"path": "a.txt", "old": "before", "new": "after"}),
            )]),
            text("done"),
        ]);
        let mut agent = Agent::new(model, CodingTools::new(dir.clone()));

        assert_eq!(agent.run("rename it", &mut |_| {}).await.unwrap(), "done");
        assert_eq!(
            tokio::fs::read_to_string(dir.join("a.txt")).await.unwrap(),
            "after\n"
        );
        // system, user, assistant, tools, assistant, tools, assistant
        assert_eq!(agent.messages().len(), 7);
    }

    #[tokio::test]
    async fn turns_tool_errors_into_results() {
        let dir = temp_dir("agent-tool-error");
        let model = Scripted::new(vec![
            calls(vec![call("call_1", "read", json!({"path": "missing.txt"}))]),
            text("gave up"),
        ]);
        let mut agent = Agent::new(model, CodingTools::new(dir));

        let mut seen = Seen::default();
        assert_eq!(
            agent.run("read it", &mut seen.callback()).await.unwrap(),
            "gave up"
        );

        let result = seen
            .0
            .iter()
            .find(|event| event.starts_with("result:"))
            .unwrap();
        assert!(result.contains("error:"), "{result}");
    }

    #[tokio::test]
    async fn stops_at_the_configured_step_limit() {
        let dir = temp_dir("agent-max-steps");
        let model = Scripted::new(vec![
            calls(vec![call("call_1", "exec", json!({"command": "true"}))]),
            calls(vec![call("call_2", "exec", json!({"command": "true"}))]),
        ]);
        let mut agent = Agent::new(model, CodingTools::new(dir)).max_iterations(2);

        let error = agent.run("loop forever", &mut |_| {}).await.unwrap_err();
        assert!(error.to_string().contains("2 steps"), "{error}");
    }

    #[tokio::test]
    async fn seeds_the_system_prompt_once_and_can_be_cleared() {
        let dir = temp_dir("agent-prompt");
        let model = Scripted::new(vec![text("one"), text("two")]);
        let mut agent = Agent::new(model, CodingTools::new(dir)).system_prompt("custom");

        agent.run("first", &mut |_| {}).await.unwrap();
        // Second run: same conversation, so no second system message.
        agent.run("second", &mut |_| {}).await.unwrap();
        let systems = agent
            .messages()
            .iter()
            .filter(|message| matches!(message, Message::System(_)))
            .count();
        assert_eq!(systems, 1);

        agent.messages_mut().push(Message::User("injected".into()));
        // system, user, assistant, user, assistant, injected
        assert_eq!(agent.messages().len(), 6);

        agent.clear();
        assert!(agent.messages().is_empty());
    }
}
