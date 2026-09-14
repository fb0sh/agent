# mini-agent

A minimal coding agent in Rust: a model loop over four tools (`read`, `write`,
`edit`, `exec`), speaking the OpenAI-compatible Chat Completions API.

```bash
mini-agent "修复当前项目的编译错误"
```

```
task → model → text or tool calls → run tools → feed results back → repeat → final answer
```

## Build

```bash
cargo build                                   # library only
cargo build --release --features cli --bin mini-agent
```

`clap` and `dotenvy` are behind the `cli` feature, so the library depends on
neither.

## CLI

```bash
export OPENAI_API_KEY=sk-...
mini-agent "add a --verbose flag to the CLI"

# Any compatible endpoint: point --base-url at it
mini-agent --base-url https://api.deepseek.com --model deepseek-chat "check this project"
mini-agent --base-url http://localhost:11434/v1 --model qwen2.5-coder "sort the imports"
```

| Flag | Env | Default |
| --- | --- | --- |
| `--model <MODEL>` | `MINI_AGENT_MODEL` | `gpt-4o-mini` |
| `--base-url <URL>` | `MINI_AGENT_BASE_URL` | `https://api.openai.com/v1` |
| `--api-key <KEY>` | `MINI_AGENT_API_KEY` | `OPENAI_API_KEY`, or none at all |
| `-C, --cwd <DIR>` | `MINI_AGENT_CWD` | current directory |
| `--max-iterations <N>` | `MINI_AGENT_MAX_ITERATIONS` | `32` |
| `-v, --verbose` | `MINI_AGENT_VERBOSE` | off |

Priority: flag → environment → default. A `.env` in the working directory fills
in whatever is not already set (`cp .env.example .env`); it is git-ignored. With
no key at all, no `Authorization` header is sent.

The CLI uses one line for progress and never scrolls: the model's reasoning is
rewritten in place, dimmed, then the answer takes that line over. `--verbose`
adds each tool call with up to three lines of its output. Nothing else reaches
stderr, so `mini-agent "..." > answer.md` captures exactly the answer.

## Library

```
Agent<M, T, C = NoopContext>
├── M: Model             OpenAiCompatible      (any chat protocol)
├── T: Toolbox           CodingTools           (any tools)
└── C: ContextManager    NoopContext, KeepLast (what the model sees)
```

Two or three lines are enough:

```rust
use agent::{Agent, CodingTools, OpenAiCompatible};

let model = OpenAiCompatible::new(OpenAiCompatible::builder().api_key(key))?;
let mut agent = Agent::new(model, CodingTools::new(cwd));

let answer = agent.run("fix the build", &mut |event| print!("{event:?}")).await?;
```

A host that wants more:

```rust
use agent::{Agent, KeepLast, OpenAiCompatible, QueueMode};

let model = OpenAiCompatible::new(
    OpenAiCompatible::builder()
        .model("deepseek-chat")
        .api_key(key)
        .timeout(Duration::from_secs(300))
        .header("x-tenant", "acme")
        // Any field this crate does not name:
        .param("reasoning_effort", "high")
        .param("temperature", 0.2),
)?;

let mut agent = Agent::new(model, CodingTools::new(cwd).max_output(1024 * 1024))
    .context(KeepLast::new(30))
    .max_iterations(64)
    .max_tool_calls(200)
    .steering_mode(QueueMode::One)
    .follow_up_mode(QueueMode::One);

let handle = agent.handle();
tokio::spawn(async move { agent.run("fix the build", &mut render).await });

handle.steer("Stop. Inspect Cargo.toml first.")?;      // changes course now
handle.follow_up("Then add a regression test.")?;      // runs after the answer
handle.abort()?;                                       // stops it
```

`AgentHandle` is `Clone + Send + Sync`, so control can come from any task; the
agent itself stays owned by the one task that runs it.

### run, step, steer, follow_up, abort

| Call | Meaning |
| --- | --- |
| `run(task, on_event)` | `push_user` then `step` until idle. The whole task. |
| `step(on_event)` | One state advance: deliver queues, one model turn and its tools. Returns `Continue` or `Idle(answer)`, giving the host control back after every step — which is where a host puts its own budgets, checkpoints or logging. |
| `push_user(text)` | Start a task without running it. |
| `handle.steer(text)` | Interrupt the current plan: the running tool finishes, the rest of that turn's tool calls are dropped (each still gets a `cancelled` result, so the history stays valid), and the message is injected before the next model call. |
| `handle.follow_up(text)` | Run after the current task is answered, as a new user message. |
| `handle.abort()` | Cancels the model request, the running tool and the command it started. Surfaces as an `agent aborted` error. |

Queues are drained before each model call: steering first, follow-ups only once
the current task has been answered. `QueueMode::One` delivers one message per
turn, `All` delivers everything queued, each as its own message.

`agent.messages()`, `messages_mut()` and `clear()` expose the conversation, and
every message type is `Serialize`/`Deserialize`, so sessions are the host's
business.

## Tools

| Tool | Arguments | Behaviour |
| --- | --- | --- |
| `read` | `path`, `offset?`, `limit?` | scans a UTF-8 file in fixed chunks; a 500 MB single line costs one chunk of memory, not 500 MB |
| `write` | `path`, `content` | creates or fully overwrites, creating parent directories |
| `edit` | `path`, `old`, `new` | exact replacement; 0 or >1 matches is an error |
| `exec` | `command` | platform shell (`sh -c`, `cmd /C`), 120 s timeout |

`max_output` bounds the whole tool result: `exec` keeps draining stdout and
stderr to EOF and discards what it cannot keep, so a chatty command never blocks
on a full pipe. Relative paths resolve against the working directory; absolute
paths are accepted. A failing tool does not abort the run — the error text goes
back to the model, including unparseable arguments, which are reported verbatim.

## Architecture

```
main.rs        CLI, config precedence, all terminal rendering
└── lib.rs         the vocabulary (Message, ToolCall, Event, …) and the traits
    ├── agent.rs       Agent, AgentHandle, step/run, queues, limits
    ├── context.rs     NoopContext, KeepLast
    ├── llm.rs         OpenAiCompatible + OpenAiConfig: request, SSE, parsing
    └── tools/         CodingTools: read, write, edit, exec bound to one cwd
```

* The library prints nothing and owns no rendering policy: `step` reports
  `Event`s and the host decides what they look like.
* Tool calls run one at a time on purpose — `write`/`edit`/`exec` depend on each
  other's effects.
* `.param()` cannot override `model`, `messages`, `tools` or `stream`: those are
  the protocol's, and `OpenAiCompatible::new` rejects them.

## Tests

```bash
cargo test --all-features
```

Covers the tools (windows, limits, empty and giant files, timeout, draining,
bounded output), request building, stream parsing, metadata round-trip, the HTTP
layer against a throwaway server, and the runtime against scripted models and
tools: `step`, `run`, tool round trips, steering (early, mid-tool, both queue
modes, skipping without breaking the tool-call pairing), follow-ups, priority,
abort of a model request and of a running command, both limits, context
preparation, history serde and bad tool arguments. No test touches the network
beyond localhost.
