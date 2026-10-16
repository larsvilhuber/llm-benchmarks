# Status and handoff

Shared between the agents working on this repo. Update your section when you finish a
step or hand something over. Pull before you start; work on your own branch; keep edits
to your own files. Do not fork `bench_llm.py`: propose changes to it in a small PR.

## Benchmark 01: Ollama on cv-ai (CUDA) vs vLLM on mac-studio (MLX): done

Results in `benchmarks/01-ollama-cuda-vs-vllm-mlx/`. vLLM setup on mac-studio
is documented in `README-mlx.md`. Open item: restart vLLM there with
`VLLM_SERVER_DEV_MODE=1 vllm serve ... --enable-sleep-mode` to allow a true cold start
(`bench_llm.py --cold-warm` detects it automatically).

## Benchmark 02: Ollama vs vLLM, both on cv-ai: not started

Plan and GPU-sharing rules: `benchmarks/02-cv-ai-ollama-vs-vllm/README.md`.

### cv-ai agent (to do)
- [ ] Install vLLM on cv-ai (in /home/vllm); confirm `vllm serve` starts
- [ ] Decide how bf16 (about 15 GB) fits 2x12 GB: tensor parallel 2 or a quantized build; record it here
- [ ] Add `--only ollama|vllm` to `bench_llm.py` (needed to run the engines in sequence)
- [ ] Ollama pass, then stop/unload Ollama, vLLM pass; record `nvidia-smi` memory
- [ ] Commit results to `benchmarks/02-cv-ai-ollama-vs-vllm/` and fill in its README

Record here when known (agents on cv-ai and mac-studio identify themselves by hostname): vLLM version, launch command and flags, model name,
Ollama model tag, Ollama `OLLAMA_NUM_PARALLEL`.

### mac-studio agent (to do)
- [ ] Keep `README-mlx.md` current (vLLM version and launch flags, model, how to enable sleep mode)

## Known issues
- On the short prompt Ollama emits 256 tokens, vLLM 59. Unknown whether Ollama is producing
  a thinking trace or not stopping; compare output text before trusting "total" times.
