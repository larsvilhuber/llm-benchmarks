# Status and handoff

Shared between the agents working on this repo. Update your section when you finish a
step or hand something over. Pull before you start; work on your own branch; keep edits
to your own files. Do not fork `bench_llm.py`: propose changes to it in a small PR.

## Benchmark 01: Ollama on cv-ai (CUDA) vs vLLM on mac-studio (MLX): done

Results in `benchmarks/01-ollama-cuda-vs-vllm-mlx/`. vLLM setup on mac-studio
is documented in `README-mlx.md`. Open item: restart vLLM there with
`VLLM_SERVER_DEV_MODE=1 vllm serve ... --enable-sleep-mode` to allow a true cold start
(`bench_llm.py --cold-warm` detects it automatically).

## Benchmark 02: Ollama vs vLLM, both on cv-ai: first warm run done

Plan and GPU-sharing rules: `benchmarks/02-cv-ai-ollama-vs-vllm/README.md`.

### cv-ai agent (to do)
- [x] Install vLLM on cv-ai (in /home/vllm); confirm `vllm serve` starts (see `README-cuda.md`)
- [x] bf16 fits 2x12 GB with `--tensor-parallel-size 2` (7.5 GB weights per GPU)
- [x] Add `--only ollama|vllm` to `bench_llm.py` (in this PR; small change, review welcome)
- [x] Ollama pass, then vLLM pass; `nvidia-smi` memory recorded (warm only; cold/warm still to do)
- [x] Results committed to `benchmarks/02-cv-ai-ollama-vs-vllm/`

cv-ai values: vLLM 0.31.0; `vllm serve google/gemma-4-E4B-it --tensor-parallel-size 2 --gpu-memory-utilization 0.93 --max-model-len 8192 --max-num-seqs 8` with `VLLM_USE_FLASHINFER_SAMPLER=0`; Ollama model `gemma4:e4b-it-bf16`; `OLLAMA_NUM_PARALLEL` not checked (requests ran serially).

Record here when known (agents on cv-ai and mac-studio identify themselves by hostname): vLLM version, launch command and flags, model name,
Ollama model tag, Ollama `OLLAMA_NUM_PARALLEL`.

### mac-studio agent (to do)
- [ ] Keep `README-mlx.md` current (vLLM version and launch flags, model, how to enable sleep mode)

## Known issues
- On the short prompt Ollama emits 256 tokens, vLLM 57. Confirmed on cv-ai: Ollama produces a thinking
  trace (`reasoning` set, `content` empty, `finish_reason: length`); vLLM does not. Totals are not
  comparable; TTFT and tok/s are. Follow-up: make thinking consistent on both sides.
