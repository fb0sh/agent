//! Built-in [`ContextManager`] implementations.
//!
//! These decide what the model is shown each turn. A host with stricter needs
//! implements the trait itself; nothing else in the crate changes.

use anyhow::Result;

use crate::{ContextManager, Message};

/// Does nothing at all: the conversation is sent as it is. This is the default,
/// so a plain `Agent::new(model, tools)` never has to think about context.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopContext;

impl ContextManager for NoopContext {
    async fn prepare(&mut self, _messages: &mut Vec<Message>) -> Result<()> {
        Ok(())
    }
}

/// Keeps the first message (the system prompt) and the most recent `keep`
/// messages, dropping everything between them.
///
/// Trimming happens in whole groups: an assistant turn that asked for tool calls
/// is never separated from the tool results that answer it, because the protocol
/// requires the pair. The cut is also pushed back so the kept window never
/// starts with a bare tool result.
///
/// ```no_run
/// # use agent::{Agent, CodingTools, KeepLast, OpenAiCompatible};
/// # fn build(model: OpenAiCompatible, tools: CodingTools) -> Agent<OpenAiCompatible, CodingTools, KeepLast> {
/// Agent::new(model, tools).context(KeepLast::new(30))
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct KeepLast {
    keep: usize,
}

impl KeepLast {
    pub fn new(keep: usize) -> Self {
        Self { keep }
    }
}

impl ContextManager for KeepLast {
    async fn prepare(&mut self, messages: &mut Vec<Message>) -> Result<()> {
        // Leave the first message alone; it is the system prompt in practice.
        if messages.len() <= self.keep + 1 {
            return Ok(());
        }

        // Start as late as allowed, then walk back over any tool results whose
        // assistant turn would otherwise be cut off.
        let mut start = messages.len() - self.keep;
        while start > 0 && matches!(messages[start], Message::Tools(_)) {
            start -= 1;
        }
        messages.drain(1..start);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ToolCall, ToolResult};
    use serde_json::json;

    fn user(text: &str) -> Message {
        Message::User(text.into())
    }

    fn call(id: &str) -> Message {
        Message::Assistant {
            text: String::new(),
            calls: vec![ToolCall {
                id: id.into(),
                name: "read".into(),
                arguments: json!({}),
            }],
            metadata: Default::default(),
        }
    }

    fn result(id: &str) -> Message {
        Message::Tools(vec![ToolResult {
            id: id.into(),
            name: "read".into(),
            output: "ok".into(),
        }])
    }

    fn role(message: &Message) -> &'static str {
        match message {
            Message::System(_) => "system",
            Message::User(_) => "user",
            Message::Assistant { .. } => "assistant",
            Message::Tools(_) => "tools",
        }
    }

    #[tokio::test]
    async fn noop_changes_nothing() {
        let mut messages = vec![Message::System("sys".into()), user("a"), user("b")];
        let before = messages.len();
        NoopContext.prepare(&mut messages).await.unwrap();
        assert_eq!(messages.len(), before);
    }

    #[tokio::test]
    async fn keep_last_always_keeps_the_system_message() {
        let mut manager = KeepLast::new(2);
        let mut messages = vec![
            Message::System("sys".into()),
            user("1"),
            user("2"),
            user("3"),
        ];
        manager.prepare(&mut messages).await.unwrap();

        assert_eq!(messages.len(), 3);
        assert!(matches!(messages[0], Message::System(_)));
        // The last two messages, plus the system prompt.
        assert!(matches!(&messages[1], Message::User(text) if text == "2"));
        assert!(matches!(&messages[2], Message::User(text) if text == "3"));
    }

    #[tokio::test]
    async fn keep_last_never_splits_an_assistant_call_from_its_results() {
        // A window of three would start on a bare tool result, so the cut moves
        // back to the assistant turn that asked for it.
        let mut manager = KeepLast::new(3);
        // system | user | assistant(call) | tools | assistant(call) | tools
        let mut messages = vec![
            Message::System("sys".into()),
            user("go"),
            call("1"),
            result("1"),
            call("2"),
            result("2"),
        ];
        manager.prepare(&mut messages).await.unwrap();

        let roles: Vec<&str> = messages.iter().map(role).collect();
        assert_eq!(
            roles,
            ["system", "assistant", "tools", "assistant", "tools"]
        );
    }

    #[tokio::test]
    async fn keep_last_leaves_short_conversations_alone() {
        let mut manager = KeepLast::new(10);
        let mut messages = vec![Message::System("sys".into()), user("a")];
        manager.prepare(&mut messages).await.unwrap();
        assert_eq!(messages.len(), 2);
    }
}
