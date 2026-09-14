# mini-agent

A minimal coding agent in Rust. It runs in a working directory with four tools
— `read`, `write`, `edit`, `exec` — and speaks three API protocols: any
OpenAI-compatible Chat Completions endpoint, Anthropic and Gemini.

```bash
mini-agent "修复当前项目的编译错误"
```

```
task → LLM → text or tool calls → run tools → feed results back → repeat → final answer
```

## Build

```bash
cargo build --release        # binary: target/release/mini-agent
```

## Usage

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
mini-agent "add a --verbose flag to the CLI"

# DeepSeek
mini-agent --api openai --base-url https://api.deepseek.com \
           --model deepseek-chat "check this project"

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
mini-agent --api anthropic --model claude-sonnet-4-5 "fix the failing tests"

# Gemini
export GEMINI_API_KEY=...
mini-agent --api gemini --model gemini-2.5-flash "explain src/main.rs"

# Local models (llama.cpp, vLLM, Ollama, LM Studio, OpenRouter, Groq, ...)
mini-agent --api openai --base-url http://localhost:11434/v1 --model qwen2.5-coder "sort the imports"
```

Any task can be given as several words (`mini-agent fix the build`).

### Options

| Flag | Env | Default |
| --- | --- | --- |
| `--api <API>` | `MINI_AGENT_API` | `openai` |
| `--model <MODEL>` | `MINI_AGENT_MODEL` | per API |
| `--base-url <URL>` | `MINI_AGENT_BASE_URL` | per API |
| `--api-key <KEY>` | `MINI_AGENT_API_KEY` | `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` |
| `-C, --cwd <DIR>` | `MINI_AGENT_CWD` | current directory |
| `--max-iterations <N>` | `MINI_AGENT_MAX_ITERATIONS` | `32` |

Priority: CLI flag → environment variable → default.

A `.env` file in the working directory is loaded automatically at startup
(`cp .env.example .env`). It only fills in variables that are not already set,
so the precedence above still holds. `.env` is git-ignored; never commit keys.

`--api` accepts `openai` (any compatible endpoint), `anthropic` and `gemini`.

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
main.rs    CLI parsing, config precedence, wiring
└── agent.rs    the loop: chat → tool calls → results → chat …
    ├── llm.rs      internal Message/Response types + one adapter per protocol
    └── tools/      read, write, edit, exec bound to one cwd
```

* `llm.rs` is the only place that knows a protocol's JSON. Everything above it
  speaks `Message` / `Response`.
* There is no provider registry, no tool trait, no service layer: `Api` is an
  enum, `chat` is a `match`, tools are `match` arms in `Tools::execute`.
* Tool calls run one at a time on purpose — `write`/`edit`/`exec` depend on each
  other's effects.

Adding a tool: a new `tools/<name>.rs`, its `definition()`, and one arm in
`Tools::execute`. Adding a protocol: `llm.rs` only.

## Tests

```bash
cargo test
```

Covers `read` (window, missing file, bad offset), `write` (create, overwrite),
`edit` (0/1/many matches), `exec` (success, non-zero exit, timeout),
`Tools`/`clip`/`resolve`, and the agent loop against a throwaway
OpenAI-compatible HTTP server: text answer, one tool call, several rounds,
tool errors fed back, and hitting `--max-iterations`.
