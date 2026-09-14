# mini-agent

A minimal coding agent in Rust. It runs in a working directory with four tools
— `read`, `write`, `edit`, `exec` — and speaks the OpenAI-compatible Chat
Completions API, so it works with OpenAI, DeepSeek, OpenRouter, Qwen, GLM,
Moonshot, Groq, Together, Fireworks, vLLM, Ollama, LM Studio and any other
compatible endpoint.

```bash
mini-agent "修复当前项目的编译错误"
```

```
task → model → text or tool calls → run tools → feed results back → repeat → final answer
```

The package is both a library (`agent`) and a thin CLI (`mini-agent`) over it.

## Build

```bash
cargo build --release        # binary: target/release/mini-agent
```

## Usage

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
mini-agent "add a --verbose flag to the CLI"

# Any compatible endpoint: point --base-url at it
mini-agent --base-url https://api.deepseek.com --model deepseek-chat "check this project"
mini-agent --base-url http://localhost:11434/v1 --model qwen2.5-coder "sort the imports"
```

Any task can be given as several words (`mini-agent fix the build`).

### Options

| Flag | Env | Default |
| --- | --- | --- |
| `--model <MODEL>` | `MINI_AGENT_MODEL` | `gpt-4o-mini` |
| `--base-url <URL>` | `MINI_AGENT_BASE_URL` | `https://api.openai.com/v1` |
| `--api-key <KEY>` | `MINI_AGENT_API_KEY` | `OPENAI_API_KEY`, or none at all |
| `-C, --cwd <DIR>` | `MINI_AGENT_CWD` | current directory |
| `--max-iterations <N>` | `MINI_AGENT_MAX_ITERATIONS` | `32` |
| `-v, --verbose` | `MINI_AGENT_VERBOSE` | off |

Priority: CLI flag → environment variable → default. With no key at all the
request is sent without an `Authorization` header, which is what local servers
such as Ollama and LM Studio want.

A `.env` file in the working directory is loaded automatically at startup
(`cp .env.example .env`). It only fills in variables that are not already set,
so the precedence above still holds. `.env` is git-ignored; never commit keys.

## Library

Three pieces, two of them extension points:

```text
Agent<M: Model, T: Toolbox>
   ├── M: Model      OpenAiCompatible (add your own protocol here)
   └── T: Toolbox    CodingTools      (add or replace tools here)
```

```rust
use agent::{Agent, CodingTools, Event, OpenAiCompatible, OpenAiConfig};

let model = OpenAiCompatible::new(
    OpenAiCompatible::builder()
        .model("deepseek-chat")
        .base_url("https://api.deepseek.com")
        .api_key(api_key)
        .timeout(Duration::from_secs(300))
        .header("x-tenant", "acme")
        // Anything this crate does not name goes through as-is:
        .param("reasoning_effort", "high")
        .param("temperature", 0.2),
)?;

let tools = CodingTools::new(cwd)
    .exec_timeout(Duration::from_secs(1800))
    .max_output(1024 * 1024)
    .read_limit(500);

let mut agent = Agent::new(model, tools).max_iterations(64);

// The library never prints: text, reasoning and tool traffic arrive as events.
let answer = agent
    .run("fix the build", &mut |event| match event {
        Event::Text(chunk) => print!("{chunk}"),
        Event::ToolCall { name, arguments } => eprintln!("{name} {arguments}"),
        _ => {}
    })
    .await?;
```

* `OpenAiConfig` is a plain struct with chainable setters, so anything not named
  there (`top_p`, `max_completion_tokens`, `seed`, `tool_choice`,
  `response_format`, `parallel_tool_calls`, vendor extensions) is passed through
  with `.param(name, value)` and nothing has to be added to the crate.
* `api_key` is optional, so endpoints that need no key work as they are.
* A host that wants a different protocol implements `Model`; a host that wants
  different tools implements `Toolbox`. Neither the agent nor the other side
  changes.
* `agent.messages()`, `messages_mut()` and `clear()` expose the conversation, so
  sessions can be saved, restored, trimmed or fed into the model again.
* `Agent::system_prompt(...)` replaces the built-in prompt.

## Tools

| Tool | Arguments | Behaviour |
| --- | --- | --- |
| `read` | `path`, `offset?`, `limit?` (1-based, 2000 lines by default) | streams a window of a UTF-8 file, clipped to `max_output` |
| `write` | `path`, `content` | creates or fully overwrites, creating parent directories |
| `edit` | `path`, `old`, `new` | exact replacement; 0 or >1 matches is an error |
| `exec` | `command` | platform shell (`sh -c`, `cmd /C`), 120 s timeout, exit code + stdout + stderr |

Relative paths resolve against the working directory; absolute paths are
accepted, since `exec` can reach the whole filesystem anyway. A failing tool does
not abort the run: the error text goes back to the model as the tool result so it
can correct itself — including unparseable arguments, which are reported
verbatim instead of being silently replaced with `{}`.

`exec` keeps reading a runaway command after the output limit is reached and
discards the rest, rather than leaving the process blocked on a full pipe.

## Architecture

```
main.rs    CLI, config precedence, all terminal rendering
└── lib.rs      the vocabulary (Message, ToolCall, Event, …) and the two traits
    ├── agent.rs    the loop: chat → tool calls → results → chat …
    ├── llm.rs      OpenAiCompatible + OpenAiConfig: request, SSE stream, parsing
    └── tools/      CodingTools: read, write, edit, exec bound to one cwd
```

* The library prints nothing and owns no rendering policy: `Agent::run` reports
  `Event`s and the CLI decides what they look like.
* The CLI uses one line and never scrolls: the model's reasoning is rewritten in
  place (dimmed, clipped to one line wide), then the answer takes that line over.
  `--verbose` adds each tool call with up to three lines of its output. Nothing
  else reaches stderr, so `mini-agent "..." > answer.md` captures exactly the
  answer.
* Tool calls run one at a time on purpose — `write`/`edit`/`exec` depend on each
  other's effects.

## Tests

```bash
cargo test
```

Covers `read` (window, configured limit, clipping, bad offset, missing file),
`write` (create, overwrite), `edit` (0/1/many matches), `exec` (success, non-zero
exit, timeout, draining past the output limit), the toolbox dispatch, request
building (history, tools, passthrough params), stream parsing (text, reasoning,
tool arguments split across events, unparseable arguments), the HTTP layer
against a throwaway server (params, headers, auth optional, error bodies), and
the agent loop against a scripted model: tool round trips, tool errors as
results, history, step limit and system prompt. `Agent` and `Toolbox` being
generic is what makes those loop tests need no network at all.
