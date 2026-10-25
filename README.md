# llm-benchmarks

Repeatable latency/throughput comparisons of LLM serving stacks, using one
shared tool, [`bench_llm.py`](bench_llm.py) (usage in [TOOL.md](TOOL.md)), plus
real coding-agent runs with [`agent_bench.py`](agent_bench.py).
Every benchmark serves the same model through the same OpenAI-compatible
streaming API, so only the stack under test differs.

| # | Benchmark | Stacks compared | Status |
|---|---|---|---|
| 01 | [Ollama + CUDA vs vLLM + MLX](benchmarks/01-ollama-cuda-vs-vllm-mlx/) | Ollama on dual-cuda (2x12 GB NVIDIA GPUs) vs vLLM on mac-studio (Apple silicon, ~50 GB) | done |
| 02 | [Ollama vs vLLM on the same hardware](benchmarks/02-dual-cuda-ollama-vs-vllm/) | both on dual-cuda: speed, 131072 context, cold start, synthetic agent session | done |
| 03 | [Large models and coding-agent backends](benchmarks/03-vllm-over-network/) | Ollama on dual-cuda, vLLM on dual-cuda, vLLM on mac-studio as OpenCode backends | OpenCode runs done; capacity sweep and Claude Code to do |

Benchmark 01 differs in hardware as well as software, so it compares deployed
setups, not engines. Benchmark 02 isolates the engine.

## Findings so far (2026-10-25)

Consolidated from benchmarks 01-03. Numbers are copied from each benchmark's README, which
has the exact commands, raw JSON and caveats; this section only puts them side by side.

Hosts: **dual-cuda** (network hostname `cv-ai`; Linux, 2x RTX 3060 12 GB, CUDA; Ollama in a container and vLLM 0.31.0 in
`/home/vllm`) and **mac-studio** (`il-lv39-m1`, Apple silicon, 64 GB unified memory, about 51 GB
usable by vLLM; vllm-metal 0.31.0, reachable only through an SSH tunnel).

### Short answers

1. **On the same GPUs, vLLM decodes about 1.7x faster than Ollama**, and its short-prompt TTFT is
   4-6x lower. Ollama prefills long prompts about 1.7x faster (cold, uncached).
2. **For a coding agent, Ollama was the more reliable backend and vLLM the faster one.** With
   OpenCode and Gemma 4 26B-A4B QAT 4-bit: Ollama on dual-cuda passed 11/12 tasks; vLLM passed 8/12 on
   both hosts, with the same two failure modes on both. When vLLM succeeded on dual-cuda it was about 3x
   faster than Ollama (16 s vs 46 s median).
3. **Model size matters more than the engine.** Gemma 4 E4B (about 4B active, bf16) passed 1/4 as an
   OpenCode backend; the 26B-A4B MoE at 4-bit passed 11/12 on the same server.
4. **Both hosts can serve a 26B MoE at 4-bit with the full 131072-token context.** mac-studio is no
   faster than dual-cuda's Ollama for agent work (42 s vs 46 s median); its advantage would be memory for
   models dual-cuda cannot hold, which has not been tested yet.

### 1. Raw serving speed

Gemma 4 E4B bf16, `--max-tokens 256`, median of 3 runs, warm.

| | Ollama, dual-cuda | vLLM, dual-cuda | vLLM, mac-studio |
|---|---|---|---|
| TTFT, short prompts | 165-180 ms | 30-47 ms | 51-69 ms |
| Decode, one stream | 33 tok/s | 57 tok/s | 44-45 tok/s |
| 4 parallel requests, aggregate | 33 tok/s (served one at a time) | 180 tok/s | 165 tok/s |
| Source | 01, 02 | 02 | 01 |

- Engine effect, same hardware (02): vLLM 1.7x on decode. Ollama ran parallel requests one after
  another with the container's settings (`OLLAMA_NUM_PARALLEL` not checked).
- Hardware effect, same engine (02 vs 01): vLLM on dual-cuda's two 3060s decodes about 1.3x faster than
  vLLM on mac-studio for this model.

### 2. Long context, cold start

Gemma 4 E4B bf16 at 131072 context on dual-cuda (02); prompts unique, so nothing is cached.

| Prompt tokens | TTFT vLLM | TTFT Ollama | Decode vLLM | Decode Ollama |
|---|---|---|---|---|
| ~3.9k | 1.8 s | 1.6 s | 54.5 tok/s | 33.3 tok/s |
| ~61k | 39.8 s | 23.0 s | 45.7 tok/s | 28.6 tok/s |
| ~128k | 110.5 s | 63.2 s | 40.0 tok/s | 24.6 tok/s |

- Uncached prefill favours Ollama at long prompts; the cause on vLLM (Triton attention for Gemma 4's
  head sizes, TP over PCIe, batching defaults) was not investigated.
- With prefix caching, as agents use it, the gap disappears after the first turn: a 20-turn synthetic
  agent session growing to ~62k tokens took 53 s on vLLM and 54 s on Ollama with one-sentence answers
  (02). Per-turn TTFT was the same (~1.85 s); vLLM's decode advantage counts in proportion to how much
  the agent writes.
- Reload: vLLM sleep mode wakes in 3.5 s (weights kept in host RAM); a full `vllm serve` start is
  166 s. Ollama loads on first use in about 6-10 s. Sleep mode is not available on mac-studio.

### 3. Coding-agent backend (OpenCode)

`agent_bench.py`, OpenCode 1.19.3, four small tasks (fix a bug, implement a function, multi-file
rename, explain code), judged by tests or an answer key, 3 runs per task, 900 s timeout, 131072 context.

| Backend | Model | Pass | Median passing run |
|---|---|---|---|
| Ollama, dual-cuda | Gemma 4 E4B bf16 (smoke, 1 run per task) | 1/4 | |
| Ollama, dual-cuda | `gemma4:26b-a4b-it-qat` (Q4_0) | **11/12** | 46 s |
| vLLM, dual-cuda | `cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4` (W4A16) | 8/12 | **16 s** |
| vLLM, mac-studio | `mlx-community/gemma-4-26B-A4B-it-qat-4bit` (MLX) | 8/12 | 42 s |

Per task (pass, median passing run):

| Task | Ollama, dual-cuda | vLLM, dual-cuda | vLLM, mac-studio |
|---|---|---|---|
| explain | 3/3, 38 s | 3/3, 11 s | 3/3, 37 s |
| fix-bug | 3/3, 46 s | 2/3, 20 s | 2/3, 38 s |
| implement | 3/3, 48 s | 1/3, 19 s | 1/3, 55 s |
| refactor | 2/3, 78 s | 2/3, 21 s | 2/3, 44 s |

How runs failed:

- **Ollama (1 failure):** a genuine mistake; the agent renamed a function but not its callers.
- **vLLM, both hosts, empty turn (3 runs each):** right after reading the file the model ends its turn
  with about 3 output tokens and no edit, so OpenCode stops. Deterministic on dual-cuda (two runs identical
  to the token).
- **vLLM, both hosts, edit loop (1 run each):** the model copies OpenCode's line-number prefixes into
  the edit's search string, every edit fails to match, and it repeats until the timeout.

Two vLLM builds with different quantizations and kernels (CUDA AWQ, Metal MLX) fail the same way while
Ollama does not, which points at something vLLM does for Gemma 4 (chat template, the `gemma4` reasoning
or tool parser, default sampling) rather than at the 4-bit weights. **This is a hypothesis; not tested.**

`tool_check.py` (single call, round trip, streaming, tool choice) passes 4/4 on every server tried, so
it does not catch these failures; only a real agent run did.

### 4. Capacity

What is known, not yet a systematic sweep (benchmark 03 Part 1 is still to do):

| Model | dual-cuda (2x12 GB) | mac-studio (~51 GB for vLLM) |
|---|---|---|
| Gemma 4 E4B bf16, 131072 ctx | fits on both engines (vLLM TP=2: 277k tokens of KV cache) | fits (default context, 131072) |
| Gemma 4 26B-A4B 4-bit, 131072 ctx | Ollama: 15.6 GB, all on GPU. vLLM: fits only with `--language-model-only` (138k tokens of KV cache) | fits |
| Gemma 4 26B-A4B 4-bit, 262144 ctx | Ollama: 2.8 GB spills to CPU | not tried |
| Gemma 4 26B-A4B bf16 (~50 GB) | does not fit | weights fill memory, no room for KV cache |

### Practical recommendation as of now

- For an OpenCode backend today: **Ollama on dual-cuda with `gemma4:26b-a4b-it-qat` at 128k context.** It is
  the only configuration that was both reliable and acceptably fast (under a minute per small task).
- vLLM on dual-cuda is the one to watch: 3x faster per task, if its Gemma 4 failure modes can be fixed.
- mac-studio brings no speed benefit for this model; its case rests on larger models (Part 1).
- While vLLM holds dual-cuda's GPUs, Ollama users (Open WebUI) cannot load models; run one engine at a time.

### Caveats that apply throughout

- Small samples (2-3 runs per configuration; 3 runs per agent task) and small agent tasks.
- "Same model" means the same weights family and quantization level, not identical files: GGUF, AWQ and
  MLX builds differ.
- Ollama emits a thinking trace by default, vLLM's Gemma 4 does not, so output lengths in 01/02 differ;
  TTFT and tok/s are comparable, totals are not.
- Nothing was disk-cold; weights came from the OS page cache.

### Open questions and next steps

1. Find the vLLM Gemma 4 agent failure: render the prompt OpenCode sends through each server's chat
   template and compare; rerun without `--reasoning-parser`; match sampling settings to Ollama's.
2. Restart mac-studio vLLM with `--enable-prompt-tokens-details` so its cache hits are visible.
3. More runs per task (5+) and harder, multi-file tasks.
4. Capacity sweep (03 Part 1), especially models above 24 GB that only mac-studio can hold, and a
   coding-specific model (Qwen3-Coder 30B) on both.
5. Claude Code as the agent (needs `/v1/messages` on the server or a translating proxy).
6. Why vLLM's uncached prefill is slower than Ollama's at long context on dual-cuda.

## Conventions

- Model: Gemma 4 E4B instruction-tuned, full precision (bf16) on both sides in 01 and 02.
  Benchmark 03 uses Gemma 4 26B-A4B QAT at 4-bit, in each engine's own format.
  Quantized builds are always labelled as such.
- Endpoints are never hard-coded; they come from arguments or environment
  (`VLLM_URL`, `OLLAMA_HOST`).
- Each benchmark directory holds its own README (setup, exact commands, results,
  caveats) and raw `results-*.json`.
