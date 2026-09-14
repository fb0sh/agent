//! The agent runtime: one loop you can drive a step at a time, plus the control
//! channel a host uses to steer it while it runs.
//!
//! This is mechanism only. It holds no policy about rendering, retries or
//! context management: those are [`Event`]s, host code and [`ContextManager`].

use std::collections::VecDeque;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::{ContextManager, Event, Message, Model, NoopContext, Response, ToolResult, Toolbox};

/// Used unless the host sets its own with [`Agent::system_prompt`].
pub const DEFAULT_SYSTEM_PROMPT: &str = "\
You are a coding agent working in the current directory.
Use the available tools to inspect, modify, and validate the project.
Inspect relevant files before editing.
Run relevant checks after modifications.";

const DEFAULT_MAX_ITERATIONS: usize = 32;

/// What a steering message that arrived mid-turn leaves behind for the tool
/// calls of that turn which never ran.
const CANCELLED: &str = "cancelled: interrupted by steering message";

/// How many queued messages are handed over at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum QueueMode {
    /// Deliver one message per turn.
    #[default]
    One,
    /// Deliver everything queued, in order, each as its own message.
    All,
}

/// What one [`Agent::step`] did.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepOutcome {
    /// There is more to do: run another step.
    Continue,
    /// Nothing is pending — no tool work, no steering, no follow-up. The model's
    /// final answer is in the payload.
    Idle(String),
}

/// What a host may send to a running agent.
#[derive(Debug)]
enum Control {
    Steer(String),
    FollowUp(String),
    Abort,
}

/// A cloneable, `Send + Sync` way to talk to an [`Agent`] that another task is
/// running. The agent itself stays owned by that one task.
#[derive(Debug, Clone)]
pub struct AgentHandle {
    control: mpsc::UnboundedSender<Control>,
}

impl AgentHandle {
    /// Interrupt the current plan at the next safe point: the tool that is
    /// running finishes, the rest of that turn's tool calls are dropped, and the
    /// message is injected before the next model call.
    pub fn steer(&self, text: impl Into<String>) -> Result<()> {
        self.send(Control::Steer(text.into()))
    }

    /// Queue a message for after the current task reaches its final answer.
    pub fn follow_up(&self, text: impl Into<String>) -> Result<()> {
        self.send(Control::FollowUp(text.into()))
    }

    /// Stop the run: cancels the model request, the running tool and the
    /// command it started. Returns as an `agent aborted` error from the loop.
    pub fn abort(&self) -> Result<()> {
        self.send(Control::Abort)
    }

    fn send(&self, control: Control) -> Result<()> {
        self.control
            .send(control)
            .map_err(|_| anyhow::anyhow!("agent is gone"))
    }
}

pub struct Agent<M: Model, T: Toolbox, C = NoopContext> {
    model: M,
    tools: T,
    context: C,
    handle: AgentHandle,
    control: mpsc::UnboundedReceiver<Control>,
    messages: Vec<Message>,
    system_prompt: String,
    steering: VecDeque<String>,
    follow_up: VecDeque<String>,
    steering_mode: QueueMode,
    follow_up_mode: QueueMode,
    max_iterations: usize,
    max_tool_calls: Option<usize>,
    turns: usize,
    executed: usize,
}

impl<M: Model, T: Toolbox> Agent<M, T, NoopContext> {
    /// A plain agent: no context management, default prompt and limits.
    pub fn new(model: M, tools: T) -> Self {
        Self::with_context(model, tools, NoopContext)
    }
}

impl<M: Model, T: Toolbox, C: ContextManager> Agent<M, T, C> {
    pub fn with_context(model: M, tools: T, context: C) -> Self {
        let (control, receiver) = mpsc::unbounded_channel();
        Self {
            model,
            tools,
            context,
            handle: AgentHandle {
                control: control.clone(),
            },
            control: receiver,
            messages: Vec::new(),
            system_prompt: DEFAULT_SYSTEM_PROMPT.to_string(),
            steering: VecDeque::new(),
            follow_up: VecDeque::new(),
            steering_mode: QueueMode::One,
            follow_up_mode: QueueMode::One,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            max_tool_calls: None,
            turns: 0,
            executed: 0,
        }
    }

    /// Replace the context manager. Everything else, conversation included, is
    /// carried over.
    pub fn context<C2: ContextManager>(self, context: C2) -> Agent<M, T, C2> {
        let Agent {
            model,
            tools,
            handle,
            control,
            messages,
            system_prompt,
            steering,
            follow_up,
            steering_mode,
            follow_up_mode,
            max_iterations,
            max_tool_calls,
            turns,
            executed,
            context: _,
        } = self;
        Agent {
            model,
            tools,
            context,
            handle,
            control,
            messages,
            system_prompt,
            steering,
            follow_up,
            steering_mode,
            follow_up_mode,
            max_iterations,
            max_tool_calls,
            turns,
            executed,
        }
    }

    /// A handle for steering, following up and aborting from another task.
    pub fn handle(&self) -> AgentHandle {
        self.handle.clone()
    }

    /// Replace the system prompt. Set it to `""` to send no system message.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Give up after this many model turns per task.
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Give up after this many tool calls per task. Only calls that actually
    /// reach the [`Toolbox`] count, which is what makes this a resource limit.
    pub fn max_tool_calls(mut self, max_tool_calls: usize) -> Self {
        self.max_tool_calls = Some(max_tool_calls);
        self
    }

    /// How many steering messages are delivered before the next model call.
    pub fn steering_mode(mut self, mode: QueueMode) -> Self {
        self.steering_mode = mode;
        self
    }

    /// How many follow-ups are delivered once the current task is answered.
    pub fn follow_up_mode(mut self, mode: QueueMode) -> Self {
        self.follow_up_mode = mode;
        self
    }

    /// The conversation so far, so a host can save, restore, trim or inspect it.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn messages_mut(&mut self) -> &mut Vec<Message> {
        &mut self.messages
    }

    /// Forget the conversation and any queued input.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.steering.clear();
        self.follow_up.clear();
        self.turns = 0;
        self.executed = 0;
    }

    /// Start a task: seed the system prompt if needed, reset the limits, and add
    /// the user message. Queued steering is left ahead of it, because a steer is
    /// higher priority than whatever was asked before.
    pub fn push_user(&mut self, text: impl Into<String>) {
        // A new task gets a fresh budget, and the system prompt if the
        // conversation does not already carry one.
        if self.messages.is_empty() && !self.system_prompt.is_empty() {
            self.messages
                .push(Message::System(self.system_prompt.clone()));
        }
        self.turns = 0;
        self.executed = 0;
        self.messages.push(Message::User(text.into()));
    }

    /// Run one step: deliver queued input, then one model turn and whatever tool
    /// work it asks for. The host gets control back after every step.
    pub async fn step(&mut self, on_event: &mut dyn FnMut(Event<'_>)) -> Result<StepOutcome> {
        self.receive()?;
        self.deliver_queues();

        // Nothing pending: the conversation already ends with an answer.
        if !self.needs_turn() {
            return Ok(StepOutcome::Idle(self.last_answer()));
        }

        if self.turns >= self.max_iterations {
            bail!(
                "agent stopped after {} steps without a final answer",
                self.max_iterations
            );
        }
        self.turns += 1;

        // The context manager sees exactly what is about to be sent.
        self.context.prepare(&mut self.messages).await?;

        // One model turn. Control messages are handled while it is in flight:
        // steering and follow-ups are queued, an abort cancels the request.
        let response = {
            let Agent {
                model,
                tools,
                control,
                steering,
                follow_up,
                messages,
                ..
            } = &mut *self;
            let definitions = tools.definitions();
            let chat = model.chat(messages, &definitions, on_event);
            wait(chat, control, steering, follow_up).await?
        }?;

        let Response {
            text,
            tool_calls,
            metadata,
        } = response;

        let answer = text.unwrap_or_default();
        if tool_calls.is_empty() {
            self.messages.push(Message::Assistant {
                text: answer.clone(),
                calls: Vec::new(),
                metadata,
            });
            // Nothing pending means this task is answered.
            if !self.steering.is_empty() || !self.follow_up.is_empty() {
                return Ok(StepOutcome::Continue);
            }
            return Ok(StepOutcome::Idle(answer));
        }

        // Tools run one at a time on purpose: write/edit/exec depend on each
        // other's effects.
        let mut results = Vec::with_capacity(tool_calls.len());
        let mut skipping = false;
        for call in &tool_calls {
            // A steering message cancels the rest of this turn's plan. Everything
            // already asked for still gets a result, or the history would break
            // the protocol's call/result pairing.
            skipping |= !self.steering.is_empty();
            if skipping {
                on_event(Event::ToolSkipped { name: &call.name });
                results.push(ToolResult {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    output: CANCELLED.to_string(),
                });
                continue;
            }
            if self.max_tool_calls.is_some_and(|max| self.executed >= max) {
                bail!(
                    "agent stopped after reaching the tool call limit ({})",
                    self.max_tool_calls.unwrap_or_default()
                );
            }
            self.executed += 1;

            on_event(Event::ToolCall {
                name: &call.name,
                arguments: &call.arguments,
            });
            let output = {
                let Agent {
                    tools,
                    control,
                    steering,
                    follow_up,
                    ..
                } = &mut *self;
                wait(tools.execute(call), control, steering, follow_up).await?
            };
            // A failed tool is not a failed run: the model gets to read the
            // error and correct itself.
            let output = output.unwrap_or_else(|error| format!("error: {error:#}"));
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
            text: answer,
            calls: tool_calls,
            metadata,
        });
        self.messages.push(Message::Tools(results));
        Ok(StepOutcome::Continue)
    }

    /// Run a task to completion on top of [`Agent::step`].
    pub async fn run(&mut self, task: &str, on_event: &mut dyn FnMut(Event<'_>)) -> Result<String> {
        self.push_user(task);
        loop {
            match self.step(on_event).await? {
                StepOutcome::Continue => {}
                StepOutcome::Idle(answer) => return Ok(answer),
            }
        }
    }

    /// Move anything the host sent into the queues.
    fn receive(&mut self) -> Result<()> {
        while let Ok(control) = self.control.try_recv() {
            route(Some(control), &mut self.steering, &mut self.follow_up)?;
        }
        Ok(())
    }

    /// Hand queued input to the conversation, steering before follow-ups.
    fn deliver_queues(&mut self) {
        if !self.steering.is_empty() {
            for text in take(&mut self.steering, self.steering_mode) {
                self.messages.push(Message::User(text));
            }
        } else if !self.needs_turn() && !self.follow_up.is_empty() {
            // A follow-up waits for the current task to be answered, so it only
            // starts once there is nothing left to continue.
            for text in take(&mut self.follow_up, self.follow_up_mode) {
                self.messages.push(Message::User(text));
            }
        }
    }

    /// The conversation ends where the model still owes an answer.
    fn needs_turn(&self) -> bool {
        matches!(
            self.messages.last(),
            Some(Message::User(_)) | Some(Message::Tools(_))
        )
    }

    fn last_answer(&self) -> String {
        match self.messages.last() {
            Some(Message::Assistant { text, calls, .. }) if calls.is_empty() => text.clone(),
            _ => String::new(),
        }
    }
}

/// Await one piece of work — a model turn or a tool call — while staying
/// responsive to the handle: an abort cancels it, and anything else is queued
/// for the next safe point.
async fn wait<F, T>(
    work: F,
    control: &mut mpsc::UnboundedReceiver<Control>,
    steering: &mut VecDeque<String>,
    follow_up: &mut VecDeque<String>,
) -> Result<T>
where
    F: std::future::Future<Output = T>,
{
    tokio::pin!(work);
    loop {
        tokio::select! {
            value = &mut work => return Ok(value),
            control = control.recv() => route(control, steering, follow_up)?,
        }
    }
}

/// Put one control message where it belongs. `Err` means the run was aborted.
fn route(
    control: Option<Control>,
    steering: &mut VecDeque<String>,
    follow_up: &mut VecDeque<String>,
) -> Result<()> {
    match control {
        Some(Control::Steer(text)) => steering.push_back(text),
        Some(Control::FollowUp(text)) => follow_up.push_back(text),
        Some(Control::Abort) => bail!("agent aborted"),
        // The agent holds a sender of its own, so this cannot happen.
        None => {}
    }
    Ok(())
}

fn take(queue: &mut VecDeque<String>, mode: QueueMode) -> Vec<String> {
    match mode {
        QueueMode::One => queue.pop_front().into_iter().collect(),
        QueueMode::All => queue.drain(..).collect(),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;
    use std::rc::Rc;
    use std::time::{Duration, Instant};

    use serde_json::json;
    use tokio::time::sleep;

    use super::*;
    use crate::tools::temp_dir;
    use crate::{CodingTools, ToolCall, ToolDefinition};

    /// Plays back responses and records the prompts it was shown.
    struct Scripted {
        replies: RefCell<VecDeque<Response>>,
        prompts: Rc<RefCell<Vec<Vec<Message>>>>,
    }

    impl Scripted {
        fn new(replies: Vec<Response>) -> Self {
            Self {
                replies: RefCell::new(replies.into()),
                prompts: Rc::new(RefCell::new(Vec::new())),
            }
        }

        fn prompts(&self) -> Rc<RefCell<Vec<Vec<Message>>>> {
            Rc::clone(&self.prompts)
        }
    }

    impl Model for Scripted {
        async fn chat(
            &self,
            messages: &[Message],
            _tools: &[ToolDefinition],
            on_event: &mut dyn FnMut(Event<'_>),
        ) -> Result<Response> {
            self.prompts.borrow_mut().push(messages.to_vec());
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

    /// Never answers, so a request can be aborted while it is in flight.
    struct Hanging;

    impl Model for Hanging {
        async fn chat(
            &self,
            _messages: &[Message],
            _tools: &[ToolDefinition],
            _on_event: &mut dyn FnMut(Event<'_>),
        ) -> Result<Response> {
            std::future::pending::<Result<Response>>().await
        }
    }

    /// Records which tools ran, taking a moment over each so a steering message
    /// can arrive while the sequence is in flight.
    struct Slow {
        ran: Rc<RefCell<Vec<String>>>,
        delay: Duration,
    }

    impl Toolbox for Slow {
        fn definitions(&self) -> Vec<ToolDefinition> {
            Vec::new()
        }

        async fn execute(&self, call: &ToolCall) -> Result<String> {
            sleep(self.delay).await;
            self.ran.borrow_mut().push(call.name.clone());
            Ok(format!("ran {}", call.name))
        }
    }

    /// Counts how often the context manager was asked to prepare.
    struct Counting(Rc<Cell<usize>>);

    impl ContextManager for Counting {
        async fn prepare(&mut self, _messages: &mut Vec<Message>) -> Result<()> {
            self.0.set(self.0.get() + 1);
            Ok(())
        }
    }

    fn text(value: &str) -> Response {
        Response {
            text: Some(value.into()),
            ..Response::default()
        }
    }

    fn call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: json!({}),
        }
    }

    fn calls(names: &[&str]) -> Response {
        Response {
            text: None,
            tool_calls: names
                .iter()
                .enumerate()
                .map(|(index, name)| call(&format!("call_{index}"), name))
                .collect(),
            metadata: Default::default(),
        }
    }

    fn write_call(id: &str, path: &str, content: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: "write".into(),
            arguments: json!({"path": path, "content": content}),
        }
    }

    fn users(prompt: &[Message]) -> Vec<String> {
        prompt
            .iter()
            .filter_map(|message| match message {
                Message::User(text) => Some(text.clone()),
                _ => None,
            })
            .collect()
    }

    fn tool_outputs(prompt: &[Message]) -> Vec<String> {
        prompt
            .iter()
            .filter_map(|message| match message {
                Message::Tools(results) => Some(results.iter().map(|r| r.output.clone())),
                _ => None,
            })
            .flatten()
            .collect()
    }

    /// Await `run` while a second future fires at the same time, so control
    /// messages can be sent to an agent that is mid-flight in one test task.
    async fn run_while<T, F>(
        agent: &mut Agent<Scripted, T>,
        task: &str,
        control: F,
    ) -> Result<String>
    where
        T: Toolbox,
        F: std::future::Future<Output = ()>,
    {
        let mut events = |_: Event<'_>| {};
        let mut run = Box::pin(agent.run(task, &mut events));
        let (answer, ()) = tokio::join!(run.as_mut(), control);
        answer
    }

    #[tokio::test]
    async fn step_returns_the_final_answer_and_then_stays_idle() {
        let dir = temp_dir("agent-step");
        let mut agent = Agent::new(Scripted::new(vec![text("hi")]), CodingTools::new(dir));
        agent.push_user("go");

        assert_eq!(
            agent.step(&mut |_| {}).await.unwrap(),
            StepOutcome::Idle("hi".into())
        );
        // Nothing pending: stepping again reports the same answer.
        assert_eq!(
            agent.step(&mut |_| {}).await.unwrap(),
            StepOutcome::Idle("hi".into())
        );
    }

    #[tokio::test]
    async fn run_is_built_on_step() {
        let dir = temp_dir("agent-run");
        let model = Scripted::new(vec![calls(&["exec"]), text("done")]);
        let prompts = model.prompts();
        let mut agent = Agent::new(model, CodingTools::new(dir));

        assert_eq!(agent.run("go", &mut |_| {}).await.unwrap(), "done");
        // Two steps' worth of model calls, then a final assistant message.
        assert_eq!(prompts.borrow().len(), 2);
        assert!(matches!(
            agent.messages().last(),
            Some(Message::Assistant { calls, .. }) if calls.is_empty()
        ));
    }

    #[tokio::test]
    async fn round_trips_tool_calls_through_the_conversation() {
        let dir = temp_dir("agent-tools");
        let model = Scripted::new(vec![
            Response {
                text: None,
                tool_calls: vec![write_call("call_1", "a.txt", "hi")],
                metadata: Default::default(),
            },
            text("wrote it"),
        ]);
        let prompts = model.prompts();
        let mut agent = Agent::new(model, CodingTools::new(dir.clone()));

        let mut seen = Vec::new();
        let answer = agent
            .run("create a.txt", &mut |event| {
                if let Event::ToolResult { output, .. } = event {
                    seen.push(output.to_string());
                }
            })
            .await
            .unwrap();

        assert_eq!(answer, "wrote it");
        assert_eq!(
            tokio::fs::read_to_string(dir.join("a.txt")).await.unwrap(),
            "hi"
        );
        assert!(seen[0].starts_with("wrote 2 bytes"), "{seen:?}");
        // The second prompt shows the assistant turn and its result.
        let second = prompts.borrow()[1].clone();
        assert!(matches!(second[2], Message::Assistant { ref calls, .. } if calls.len() == 1));
        assert!(matches!(second[3], Message::Tools(_)));
    }

    #[tokio::test]
    async fn steering_queued_before_a_run_reaches_the_first_prompt() {
        let dir = temp_dir("agent-steer-early");
        let model = Scripted::new(vec![text("done")]);
        let prompts = model.prompts();
        let mut agent = Agent::new(model, CodingTools::new(dir));
        let handle = agent.handle();

        handle.steer("look at Cargo.toml first").unwrap();
        agent.run("fix it", &mut |_| {}).await.unwrap();

        // Queued input is appended to the conversation, then sent with it.
        let prompt = prompts.borrow()[0].clone();
        assert_eq!(users(&prompt), ["fix it", "look at Cargo.toml first"]);
    }

    #[tokio::test]
    async fn steering_mid_tools_cancels_the_rest_of_the_plan() {
        let ran = Rc::new(RefCell::new(Vec::new()));
        let model = Scripted::new(vec![calls(&["a", "b", "c"]), text("changed course")]);
        let prompts = model.prompts();
        let mut agent = Agent::new(
            model,
            Slow {
                ran: Rc::clone(&ran),
                delay: Duration::from_millis(60),
            },
        );
        let handle = agent.handle();

        let answer = run_while(&mut agent, "go", async move {
            sleep(Duration::from_millis(20)).await;
            handle.steer("stop those").unwrap();
        })
        .await
        .unwrap();

        assert_eq!(answer, "changed course");
        // The tool that was already running finished; the rest were dropped.
        assert_eq!(*ran.borrow(), ["a"]);
        // Every call still has a result, so the protocol stays valid.
        let second = prompts.borrow()[1].clone();
        let outputs = tool_outputs(&second);
        assert_eq!(outputs.len(), 3, "{outputs:?}");
        assert_eq!(outputs[0], "ran a");
        assert_eq!(outputs[1], CANCELLED);
        assert_eq!(outputs[2], CANCELLED);
        // ... and the steering message is in there too.
        assert!(users(&second).contains(&"stop those".to_string()));
    }

    #[tokio::test]
    async fn steering_mode_one_delivers_one_at_a_time() {
        let dir = temp_dir("agent-steer-one");
        let model = Scripted::new(vec![text("one"), text("two")]);
        let prompts = model.prompts();
        let mut agent = Agent::new(model, CodingTools::new(dir));
        let handle = agent.handle();

        handle.steer("A").unwrap();
        handle.steer("B").unwrap();
        let answer = agent.run("task", &mut |_| {}).await.unwrap();

        assert_eq!(answer, "two");
        let first = prompts.borrow()[0].clone();
        let second = prompts.borrow()[1].clone();
        assert_eq!(users(&first), ["task", "A"]);
        assert_eq!(users(&second), ["task", "A", "B"]);
    }

    #[tokio::test]
    async fn steering_mode_all_delivers_everything_at_once() {
        let dir = temp_dir("agent-steer-all");
        let model = Scripted::new(vec![text("done")]);
        let prompts = model.prompts();
        let mut agent = Agent::new(model, CodingTools::new(dir)).steering_mode(QueueMode::All);
        let handle = agent.handle();

        handle.steer("A").unwrap();
        handle.steer("B").unwrap();
        assert_eq!(agent.run("task", &mut |_| {}).await.unwrap(), "done");

        let first = prompts.borrow()[0].clone();
        assert_eq!(users(&first), ["task", "A", "B"]);
    }

    #[tokio::test]
    async fn a_follow_up_runs_after_the_current_task_is_answered() {
        let dir = temp_dir("agent-follow-up");
        let model = Scripted::new(vec![text("first"), text("second")]);
        let prompts = model.prompts();
        let mut agent = Agent::new(model, CodingTools::new(dir));
        let handle = agent.handle();

        handle.follow_up("now add tests").unwrap();
        let answer = agent.run("fix it", &mut |_| {}).await.unwrap();

        assert_eq!(answer, "second");
        let first = prompts.borrow()[0].clone();
        let second = prompts.borrow()[1].clone();
        // The follow-up is not in the first prompt: the task finishes first.
        assert_eq!(users(&first), ["fix it"]);
        assert_eq!(users(&second), ["fix it", "now add tests"]);
    }

    #[tokio::test]
    async fn follow_up_mode_all_delivers_everything_at_once() {
        let dir = temp_dir("agent-follow-up-all");
        let model = Scripted::new(vec![text("first"), text("second")]);
        let prompts = model.prompts();
        let mut agent = Agent::new(model, CodingTools::new(dir)).follow_up_mode(QueueMode::All);
        let handle = agent.handle();

        handle.follow_up("one").unwrap();
        handle.follow_up("two").unwrap();
        assert_eq!(agent.run("task", &mut |_| {}).await.unwrap(), "second");

        assert_eq!(users(&prompts.borrow()[1]), ["task", "one", "two"]);
    }

    #[tokio::test]
    async fn steering_is_delivered_before_follow_ups() {
        let dir = temp_dir("agent-priority");
        let model = Scripted::new(vec![text("one"), text("two")]);
        let prompts = model.prompts();
        let mut agent = Agent::new(model, CodingTools::new(dir));
        let handle = agent.handle();

        handle.follow_up("later").unwrap();
        handle.steer("now").unwrap();
        assert_eq!(agent.run("task", &mut |_| {}).await.unwrap(), "two");

        // Steering first, then the follow-up once the task is answered.
        assert_eq!(users(&prompts.borrow()[0]), ["task", "now"]);
        assert_eq!(users(&prompts.borrow()[1]), ["task", "now", "later"]);
    }

    #[tokio::test]
    async fn abort_cancels_an_in_flight_model_request() {
        let dir = temp_dir("agent-abort-model");
        let mut agent = Agent::new(Hanging, CodingTools::new(dir));
        let handle = agent.handle();

        let started = Instant::now();
        let mut events = |_: Event<'_>| {};
        let mut run = Box::pin(agent.run("go", &mut events));
        let control = async move {
            sleep(Duration::from_millis(30)).await;
            handle.abort().unwrap();
        };
        let (answer, ()) = tokio::join!(run.as_mut(), control);
        let error = answer.unwrap_err().to_string();

        assert!(error.contains("aborted"), "{error}");
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "did not abort promptly"
        );
    }

    #[tokio::test]
    async fn abort_stops_a_running_command() {
        let dir = temp_dir("agent-abort-exec");
        let model = Scripted::new(vec![Response {
            text: None,
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "exec".into(),
                arguments: json!({"command": "sleep 30; echo x > marker"}),
            }],
            metadata: Default::default(),
        }]);
        let mut agent = Agent::new(model, CodingTools::new(dir.clone()));
        let handle = agent.handle();

        let started = Instant::now();
        let mut events = |_: Event<'_>| {};
        let mut run = Box::pin(agent.run("go", &mut events));
        let control = async move {
            sleep(Duration::from_millis(50)).await;
            handle.abort().unwrap();
        };
        let (answer, ()) = tokio::join!(run.as_mut(), control);
        let error = answer.unwrap_err().to_string();

        assert!(error.contains("aborted"), "{error}");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the command was not stopped"
        );
        assert!(!dir.join("marker").exists());
    }

    #[tokio::test]
    async fn stops_at_the_iteration_limit() {
        let dir = temp_dir("agent-max-iterations");
        let model = Scripted::new(vec![calls(&["a"]), calls(&["b"])]);
        let mut agent = Agent::new(model, CodingTools::new(dir)).max_iterations(2);

        let error = agent.run("go", &mut |_| {}).await.unwrap_err();
        assert!(error.to_string().contains("2 steps"), "{error}");
    }

    #[tokio::test]
    async fn stops_at_the_tool_call_limit() {
        let dir = temp_dir("agent-max-tools");
        let model = Scripted::new(vec![calls(&["a", "b"]), text("done")]);
        let mut agent = Agent::new(model, CodingTools::new(dir)).max_tool_calls(1);

        let error = agent.run("go", &mut |_| {}).await.unwrap_err();
        assert!(error.to_string().contains("tool call limit"), "{error}");
    }

    #[tokio::test]
    async fn context_is_prepared_before_every_model_call() {
        let dir = temp_dir("agent-context");
        let calls = Rc::new(Cell::new(0));
        let model = Scripted::new(vec![
            Response {
                text: None,
                tool_calls: vec![write_call("call_1", "a.txt", "hi")],
                metadata: Default::default(),
            },
            text("done"),
        ]);
        let mut agent =
            Agent::with_context(model, CodingTools::new(dir), Counting(Rc::clone(&calls)));

        agent.run("go", &mut |_| {}).await.unwrap();
        assert_eq!(calls.get(), 2);
    }

    #[tokio::test]
    async fn history_survives_a_serde_round_trip() {
        let dir = temp_dir("agent-serde");
        let model = Scripted::new(vec![
            Response {
                text: None,
                tool_calls: vec![write_call("call_1", "a.txt", "hi")],
                metadata: Default::default(),
            },
            text("done"),
        ]);
        let mut agent = Agent::new(model, CodingTools::new(dir));
        agent.run("go", &mut |_| {}).await.unwrap();

        let saved = serde_json::to_string(agent.messages()).unwrap();
        let restored: Vec<Message> = serde_json::from_str(&saved).unwrap();
        assert_eq!(serde_json::to_string(&restored).unwrap(), saved);

        // A restored conversation can be handed to a fresh agent.
        let mut next = Agent::new(
            Scripted::new(vec![text("again")]),
            CodingTools::new(temp_dir("agent-serde-2")),
        );
        next.messages_mut().extend(restored);
        assert_eq!(next.run("more", &mut |_| {}).await.unwrap(), "again");
    }

    #[tokio::test]
    async fn unparseable_tool_arguments_reach_the_tool() {
        let dir = temp_dir("agent-bad-args");
        let model = Scripted::new(vec![
            Response {
                text: None,
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "read".into(),
                    // What the wire layer keeps when the model sends bad JSON.
                    arguments: serde_json::Value::String("{\"path\":".into()),
                }],
                metadata: Default::default(),
            },
            text("gave up"),
        ]);
        let mut agent = Agent::new(model, CodingTools::new(dir));

        let mut results = Vec::new();
        agent
            .run("go", &mut |event| {
                if let Event::ToolResult { output, .. } = event {
                    results.push(output.to_string());
                }
            })
            .await
            .unwrap();

        assert!(results[0].contains("error:"), "{results:?}");
        assert!(results[0].contains("{\\\"path\\\":"), "{results:?}");
    }
}
