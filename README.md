# llm-benchmarks

Repeatable latency/throughput comparisons of LLM serving stacks, using one
shared tool, [`bench_llm.py`](bench_llm.py) (usage in [TOOL.md](TOOL.md)), plus
real coding-agent runs with [`agent_bench.py`](agent_bench.py).
Every benchmark serves the same model through the same OpenAI-compatible
streaming API, so only the stack under test differs.

**Results across all benchmarks: [FINDINGS.md](FINDINGS.md).**

| # | Benchmark | Stacks compared | Status |
|---|---|---|---|
| 01 | [Ollama + CUDA vs vLLM + MLX](benchmarks/01-ollama-cuda-vs-vllm-mlx/) | Ollama on cv-ai (2x12 GB NVIDIA GPUs) vs vLLM on mac-studio (Apple silicon, ~50 GB) | done |
| 02 | [Ollama vs vLLM on the same hardware](benchmarks/02-cv-ai-ollama-vs-vllm/) | both on cv-ai: speed, 131072 context, cold start, synthetic agent session | done |
| 03 | [Large models and coding-agent backends](benchmarks/03-vllm-over-network/) | Ollama on cv-ai, vLLM on cv-ai, vLLM on mac-studio as OpenCode backends | OpenCode runs done; capacity sweep and Claude Code to do |

Benchmark 01 differs in hardware as well as software, so it compares deployed
setups, not engines. Benchmark 02 isolates the engine.

## Conventions

- Model: Gemma 4 E4B instruction-tuned, full precision (bf16) on both sides in 01 and 02.
  Benchmark 03 uses Gemma 4 26B-A4B QAT at 4-bit, in each engine's own format.
  Quantized builds are always labelled as such.
- Endpoints are never hard-coded; they come from arguments or environment
  (`VLLM_URL`, `OLLAMA_HOST`).
- Each benchmark directory holds its own README (setup, exact commands, results,
  caveats) and raw `results-*.json`.
