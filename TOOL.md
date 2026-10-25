# LLM endpoint benchmark: vLLM vs Ollama

`bench_llm.py` compares latency and throughput of a vLLM server and an Ollama
server serving the same model. Both are driven through the same streaming
OpenAI-compatible `/v1/chat/completions` API, so the measurements are
like-for-like. Python 3 standard library only; no dependencies.

## What it measures

For each prompt (short, medium, long), after a warmup request:

- **TTFT**: time to first streamed token (median of `--runs`)
- **tok/s**: decode throughput after the first token (median)
- **total**: end-to-end time and output token count
- **parallel test**: `--concurrency` simultaneous requests, reporting aggregate tok/s
- a final vLLM-vs-Ollama speedup table

Sampling is deterministic (`temperature=0`). Token counts come from the server's
`usage` field when available, otherwise from streamed chunk counts.

## Prerequisites

1. **vLLM** is serving the model (here: `vllm serve google/gemma-4-E4B-it --host 0.0.0.0 --port 8000` on mac-studio).
2. **Ollama** has a build with matching weights. vLLM serves full precision, so
   compare against the bf16 Ollama build, not the default 4-bit one:
   ```
   ollama pull gemma4:e4b-it-bf16        # on the Ollama host (~16 GB)
   ```
   The default `gemma4:e4b` is Q4_K_M, so results would mix engine speed with quantization.
3. **SSH key access** to the vLLM host (the tunnel runs with `BatchMode`, so no password prompts).

## Configuration (no hard-coded endpoints)

| Setting | Argument | Environment fallback |
|---|---|---|
| vLLM base URL | `--vllm-url` | `VLLM_URL` |
| Ollama base URL | `--ollama-url` | `OLLAMA_URL`, then `OLLAMA_HOST` |
| Ollama model | `--ollama-model` (required) | none |
| vLLM model | `--vllm-model` | first model from the server's `/v1/models` |

Both URLs must be set; the script exits with an error if either is missing.

## Repeat run

```
export VLLM_URL=http://mac-studio:8000     # OLLAMA_HOST (cv-ai:11434) must also be set
./bench_llm.py --ssh-tunnel --ollama-model gemma4:e4b-it-bf16 --wait 300 --json results.json
```

- `--ssh-tunnel` opens `ssh -N -L <free local port>:127.0.0.1:<port> <host>`, taking
  host and port from `--vllm-url`, sends vLLM traffic through it, and closes it on exit.
  Omit it if the vLLM port is directly reachable.
- `--wait N` polls both servers for up to N seconds, so you can start the run while a
  server is still loading or downloading.
- `--json FILE` saves raw results.

## Cold vs warm run

```
./bench_llm.py --ssh-tunnel --cold-warm --ollama-model gemma4:e4b-it-bf16 --json results-cold-warm.json
```

No separate warmup request is sent in this mode. Per engine:

1. **Ollama:** the model is unloaded (`keep_alive: 0`, confirmed via `/api/ps`), so the
   **cold** pass includes the model load in its first TTFT. The **warm** pass repeats the
   same load with the model already in memory. The model is unloaded again afterwards.
2. **vLLM:** the server holds the model in memory and has no unload call. If it was
   started with sleep mode (`VLLM_SERVER_DEV_MODE=1 vllm serve ... --enable-sleep-mode`),
   the script calls `/sleep`, times `/wake_up` (the reload), and runs the cold pass.
   Otherwise it says so and the "cold" pass is just the first pass on the running
   server, which still shows one-time costs (kernel compilation, caches) but no weight reload.
   vLLM is left awake afterwards so the server stays usable.

The speedup table is printed per phase (cold and warm).

## Useful options

| Option | Default | Purpose |
|---|---|---|
| `--prompts short medium long` | all | which prompts to run |
| `--runs N` | 3 | repetitions per prompt (median reported) |
| `--max-tokens N` | 256 | output cap per request |
| `--concurrency N` | 4 | parallel-request test; `1` disables |

## Caveats

- Keep the model state the same on both sides: nothing else loaded on the Ollama
  GPU, vLLM idle. The first Ollama request after an idle period loads the model;
  the warmup request absorbs this.
- Gemma 4 supports thinking mode. If one server emits a thinking trace and the
  other does not, output token counts differ and per-request totals are not comparable;
  tok/s and TTFT remain comparable.
- Ollama's default context window is smaller than the model maximum; the short
  prompts here do not hit it.
- vLLM on macOS (Metal) performance differs from CUDA; results are specific to this hardware.

## One engine at a time, long context, agent sessions

| Option | Purpose |
|---|---|
| `--only ollama\|vllm` | benchmark one engine (when both share the GPUs and must run in sequence); only that URL is needed |
| `--context-tokens N ...` | instead of `--prompts`, filler prompts of about N tokens, unique per request (no prefix-cache hits); prints the server's prompt token count |
| `--agent-turns N` | N-turn coding-agent-shaped session: fixed system prompt, a tool output appended each turn, full history resent (prefix cache can hit); prints cached tokens per turn |
| `--agent-prefix T` / `--agent-step S` | agent mode: system prompt size (default 15000) / tokens appended per turn (default 1500) |
| `--agent-interleave T` | agent mode: send an unrelated T-token request between turns (another user on the server) |
| `--no-think` | send `reasoning_effort: "none"` (turns off Gemma 4 thinking on Ollama; vLLM's default) |

Example (vLLM must report cached tokens: `--enable-prompt-tokens-details`):

```
./bench_llm.py --only vllm --ollama-model x --agent-turns 20 --agent-prefix 15000 --agent-step 2500 \
  --max-tokens 256 --no-think --json results-agent-vllm.json
```

`--ollama-model` is still required by the parser; with `--only vllm` its value is unused.

## Tool-calling check

`./tool_check.py --url <base URL> [--model M] [--json out.json]` checks the OpenAI tool-calling path an agent
uses: a tool call, a round trip with the tool result, a streamed tool call, and choosing between two tools.
Exit code 0 when all pass.
