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

cv-ai max-context run (131072 tokens, single user): done, results in the benchmark 02 README. Ollama tag
`gemma4:e4b-it-bf16-ctx128k` (num_ctx 131072) was created on cv-ai for it; vLLM used
`--max-model-len 131072 --max-num-seqs 1`. Open: vLLM prefill is slower than Ollama at long context (cause not investigated).

Cold-start run on cv-ai done (vLLM with `VLLM_SERVER_DEV_MODE=1 --enable-sleep-mode`; `vllm serve` start to ready 166 s).

Coding-agent run on cv-ai done: `tool_check.py` passes on both engines (vLLM needs
`--enable-auto-tool-choice --tool-call-parser gemma4`); 20-turn cached session takes the same time on both
with short answers, vLLM ahead when answers are longer (decode 1.7x). Not done: a real OpenCode session,
a larger coding model (e.g. qwen3-coder:30b, 4-bit on vLLM).

## Known issues
- On the short prompt Ollama emits 256 tokens, vLLM 57. Confirmed on cv-ai: Ollama produces a thinking
  trace (`reasoning` set, `content` empty, `finish_reason: length`); vLLM does not. Totals are not
  comparable; TTFT and tok/s are. Follow-up: make thinking consistent on both sides.

## OpenCode workflow (branch `opencode-workflow`, stacked on PR #1)

`agent_bench.py` + `tasks/` run a real OpenCode session against any OpenAI-compatible server and
judge it by outcome (see "Coding-agent workflow" in `TOOL.md`). Validated once against Ollama on cv-ai
(`gemma4:e4b-it-bf16-ctx128k`, 1 run per task, smoke test only,
`benchmarks/03-vllm-over-network/results-opencode-ollama-smoke.json`): implement passed (87 s);
explain failed (wrong answer); refactor failed (stopped after one step with no tool calls); fix-bug
timed out at 400 s after 10 tool calls. Not yet run against vLLM on either host (cv-ai's vLLM binds
127.0.0.1; mac-studio's was not answering).

Gotchas found: OpenCode takes its directory from `$PWD`, not the process cwd, so a harness must set
`PWD` (done in `agent_bench.py`). Ollama here serves one request at a time, so a killed agent run can
leave a generation running that delays the next run; mind this when other agents use cv-ai.
Next: 5 runs per task on each target, a larger coding model (e.g. qwen3-coder), and vLLM on both hosts.
