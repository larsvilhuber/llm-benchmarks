# llm-benchmarks

Repeatable latency/throughput comparisons of LLM serving stacks, using one
shared tool, [`bench_llm.py`](bench_llm.py) (usage in [TOOL.md](TOOL.md)).
Every benchmark serves the same model through the same OpenAI-compatible
streaming API, so only the stack under test differs.

| # | Benchmark | Stacks compared | Status |
|---|---|---|---|
| 01 | [Ollama + CUDA vs vLLM + MLX](benchmarks/01-ollama-cuda-vs-vllm-mlx/) | Ollama on 2x12 GB NVIDIA GPUs vs vLLM on Apple silicon (~50 GB) | done |
| 02 | [Ollama vs vLLM on the same hardware](benchmarks/02-cv-ai-ollama-vs-vllm/) | both on the 2x12 GB CUDA host | planned (needs vLLM installed) |

Benchmark 01 differs in hardware as well as software, so it compares deployed
setups, not engines. Benchmark 02 isolates the engine.

## Conventions

- Model: Gemma 4 E4B instruction-tuned, full precision (bf16) on both sides.
  Quantized builds are a separate run, labelled as such.
- Endpoints are never hard-coded; they come from arguments or environment
  (`VLLM_URL`, `OLLAMA_HOST`).
- Each benchmark directory holds its own README (setup, exact commands, results,
  caveats) and raw `results-*.json`.
