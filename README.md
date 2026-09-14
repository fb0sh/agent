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
task → LLM → text or tool calls → run tools → feed results back → repeat → final answer
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

# Any compatible endpoint: just point --base-url at it
mini-agent --base-url https://api.deepseek.com --model deepseek-chat "check this project"
mini-agent --base-url https://openrouter.ai/api/v1 --model qwen/qwen3-coder "sort the imports"
mini-agent --base-url http://localhost:11434/v1 --model qwen2.5-coder "explain src/main.rs"
```

Any task can be given as several words (`mini-agent fix the build`).

### Options

| Flag | Env | Default |
| --- | --- | --- |
| `--model <MODEL>` | `MINI_AGENT_MODEL` | `gpt-4o-mini` |
| `--base-url <URL>` | `MINI_AGENT_BASE_URL` | `https://api.openai.com/v1` |
| `--api-key <KEY>` | `MINI_AGENT_API_KEY` | `OPENAI_API_KEY` |
| `-C, --cwd <DIR>` | `MINI_AGENT_CWD` | current directory |
| `--max-iterations <N>` | `MINI_AGENT_MAX_ITERATIONS` | `32` |
| `-v, --verbose` | `MINI_AGENT_VERBOSE` | off |

Priority: CLI flag → environment variable → default.

A `.env` file in the working directory is loaded automatically at startup
(`cp .env.example .env`). It only fills in variables that are not already set,
so the precedence above still holds. `.env` is git-ignored; never commit keys.

## Library

```rust
use agent::{Agent, Llm, Output, Tools};

let llm = Llm::new(Some("deepseek-chat".into()), None, api_key)?;
let mut agent = Agent::new(llm, Tools::new(cwd), 32, Output::Quiet);

// Text and reasoning arrive as they are generated; the return value is the
// final answer. `Output::Quiet` means the library itself prints nothing.
let answer = agent.run("fix the build", &mut |delta, chunk| print!("{chunk}")).await?;
```

`llm.rs` is the only module that knows the wire format, `agent.rs` is the loop,
`tools/` is the four tools bound to one working directory. Adding a tool means a
new `tools/<name>.rs`, its `definition()`, and one arm in `Tools::execute`.

## Tools

| Tool | Arguments | Behaviour |
| --- | --- | --- |
| `read` | `path`, `offset?`, `limit?` (1-based, 2000 lines by default) | streams a window of a UTF-8 file, output clipped at 100 KB |
| `write` | `path`, `content` | creates or fully overwrites, creating parent directories |
| `edit` | `path`, `old`, `new` | exact replacement; 0 or >1 matches is an error |
| `exec` | `command` | `sh -c` in the working directory, 120 s timeout, exit code + stdout + stderr |

A failing tool does not abort the run: the error text goes back to the model as
the tool result so it can correct itself.

## Architecture

```
main.rs    CLI, config precedence, rendering
└── lib.rs      the library: agent, llm, tools
    ├── agent.rs    the loop: chat → tool calls → results → chat …
    ├── llm.rs      OpenAI-compatible protocol: request, SSE stream, parsing
    └── tools/      read, write, edit, exec bound to one cwd
```

* The library never writes to the terminal. Text and reasoning arrive through
  the callback given to `Agent::run`; progress output is opt-in via `Output`
  (`Quiet` / `Progress` / `Verbose`), and the CLI picks it.
* Progress uses one line and no scrolling: the model's reasoning rewritten in
  place (dimmed, clipped to one line wide), then the answer taking that line
  over. Nothing else reaches stderr, so `mini-agent "..." > answer.md` captures
  exactly the answer.
* `--verbose` also prints each tool call with up to three lines of its output.
  Note that some gateways stream only the reasoning and deliver the answer text
  in one burst at the end — the reasoning line is the live part there.
* No provider registry, no tool trait, no service layer: `chat` is one path,
  tools are `match` arms in `Tools::execute`.
* Tool calls run one at a time on purpose — `write`/`edit`/`exec` depend on each
  other's effects.

## Tests

```bash
cargo test
```

Covers `read` (window, missing file, bad offset), `write` (create, overwrite),
`edit` (0/1/many matches), `exec` (success, non-zero exit, timeout),
`Tools`/`clip`/`resolve`, request building, stream parsing (text, reasoning and
tool arguments split across events), and the agent loop against a throwaway
OpenAI-compatible SSE server: text answer, streaming deltas, reasoning kept out
of the answer, reassembly of split tool arguments, one tool call, several rounds,
tool errors fed back, and hitting `--max-iterations`.
