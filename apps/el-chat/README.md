# el-chat — local LLM chat test client

An interactive command-line client that holds a multi-turn conversation with a
small LLM running **entirely on your machine**. It exists to exercise the Edge
Intelligence SDK end-to-end: every reply flows through the SDK's public seams,
and the client itself depends only on SDK crates (`el-core`, `el-engine-candle`)
— it contains no inference, model, or tokenizer code of its own.

```
el-chat  →  el_core::LlmProvider  →  el_engine_candle::QwenChatProvider
                                       (real Qwen2 forward via candle-transformers)
                                  →  el_runtime::InferenceSession
                                       (provenance gate → prefill → decode loop)
```

---

## 1. Prerequisites

- **Rust 1.96+** (matches the workspace `rust-version`).
- A **local GGUF model** of the Qwen2 family plus its `tokenizer.json`
  (downloaded once, see below). Nothing is fetched at runtime — the client runs
  fully offline / air-gapped (ADR-004).

## 2. Get a model (once)

From the **repository root**, download a small instruct model (~470 MB) and its
tokenizer into a git-ignored `models/` directory:

```sh
mkdir -p models

curl -sSL -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf

curl -sSL -o models/qwen2.5-0.5b-instruct.tokenizer.json \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json
```

These two paths are the client's defaults, so no flags are needed if you place
the files here. Any other Qwen2-family GGUF works too — pass `--model` /
`--tokenizer` to point elsewhere.

## 3. Run

Interactive REPL (run from the repository root):

```sh
cargo run -p el-chat
```

You'll see a prompt; type a message and press Enter:

```
you> What is the capital of France?
bot> The capital of France is Paris.
you> What language do they speak there?
bot> They speak French in France.
you> /exit
```

The second answer shows that context carries across turns.

One-shot, non-interactive (handy for scripts and quick checks):

```sh
# Reply to a single message and exit:
cargo run -p el-chat -- --prompt "Explain a mutex in one sentence." --once

# Or pipe the message in on stdin:
echo "List three primary colors." | cargo run -p el-chat -- --once
```

> First run compiles the ML dependencies (a few minutes). Subsequent runs start
> in well under a second. The model itself loads in ~0.5 s.

## 4. Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m`, `--model <PATH>` | `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` | GGUF model file |
| `-t`, `--tokenizer <PATH>` | `models/qwen2.5-0.5b-instruct.tokenizer.json` | `tokenizer.json` |
| `-s`, `--system <TEXT>` | "You are a helpful, concise assistant…" | System prompt |
| `-p`, `--prompt <TEXT>` | — | Send one message, print the reply, exit |
| `--once` | — | Read one line from stdin, reply, exit |
| `--max-tokens <N>` | `512` | Max tokens generated per reply |
| `-h`, `--help` | — | Show help |

Example with a custom persona and shorter replies:

```sh
cargo run -p el-chat -- \
  --system "You are a terse pirate. Answer in one sentence." \
  --max-tokens 80
```

## 5. REPL commands

Type these instead of a message:

| Command | Effect |
|---------|--------|
| `/reset` | Clear the conversation (keep the current system prompt) |
| `/system <text>` | Replace the system prompt and start a fresh conversation |
| `/help` | Show usage |
| `/exit` (`/quit`, `/q`) | Leave the client (Ctrl-D also works) |

## 6. How it works

The client builds a `Vec<ChatMessage>` (a system message plus the running
conversation) and, each turn, hands the whole list to the SDK as a
`ChatRequest`. `QwenChatProvider` renders it to Qwen2.5 ChatML, tokenizes it,
and runs the standard `el_runtime::InferenceSession`:

1. a **provenance `LoadPermit`** is required before the model can load (ADR-006);
2. **prefill** feeds the prompt into the engine's KV cache;
3. the **decode loop** runs `grammar mask → safety steer → commit` per token.

Replies print through the SDK's `LlmProvider::chat_stream`.

## 7. Notes and limitations

- **Deterministic output.** The SDK runtime decodes with greedy argmax — there is
  no temperature/top-p sampling on the local path — so the same prompt always
  produces the same reply. (Cloud backends, via `el-cloud`, do honor
  temperature.)
- **Per-turn cost grows.** Each turn rebuilds the session and re-processes the
  whole conversation, so later turns in a long chat take longer. Use `/reset`
  to start fresh. Keep `--max-tokens` modest for snappier replies.
- **Small model.** Qwen2.5-0.5B is fast and runs anywhere, but it is a 0.5B
  model — expect occasional mistakes. Larger Qwen2 GGUFs improve quality at the
  cost of speed and memory.
- **CPU only** in the default build. Replies generate at a few tokens/second on
  a typical laptop CPU.

## 8. Troubleshooting

- **`model file not found`** — you haven't downloaded the model, or you're not
  running from the repository root. Re-check step 2 or pass `--model` /
  `--tokenizer` with explicit paths.
- **`failed to load tokenizer.json`** — the tokenizer path is wrong or the file
  is corrupt; re-download it.
- **`GGUF: failed to load Qwen2 weights`** — the file isn't a Qwen2-family GGUF
  (this client uses the Qwen2 architecture). Use a Qwen2/Qwen2.5 GGUF.
- **Garbled or repetitive output** — make sure the model and tokenizer come from
  the *same* model family/version.
