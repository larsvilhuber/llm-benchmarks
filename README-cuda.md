# Serving Gemma 4 with vLLM on CUDA (dual-cuda, 2x RTX 3060 12 GB)

Notes from installing vLLM 0.31.0 on dual-cuda (Linux, 2x NVIDIA GeForce RTX 3060 12 GB,
driver 580.178.04 / CUDA 13.0, 24 cores, 30 GB RAM, PCIe only, no NVLink) and serving
`google/gemma-4-E4B-it` in bf16 across both GPUs. Companion to [README-mlx.md](README-mlx.md).

## Install

Everything lives under `/home/vllm`, in its own virtualenv. The wheel bundles its CUDA
runtime (torch 2.13.0+cu130), so no system CUDA toolkit is needed.

```bash
cd /home/vllm
uv venv --python 3.12 vllm-env
uv pip install --python vllm-env/bin/python vllm      # vllm 0.31.0, big download (~10 min)
vllm-env/bin/python -c "import vllm, torch; print(vllm.__version__, torch.cuda.device_count())"
# 0.31.0 2
```

## Hugging Face token

Gemma is gated: accept the license on the model page and export a token before serving.
The token is kept outside the repo in `~/.envvars` (must define `HF_TOKEN`). Never commit it.
The weights (~15 GB) are cached inside `/home/vllm` with `HF_HOME=/home/vllm/hf-cache`.

## Launch command that works

```bash
cd /home/vllm && mkdir -p logs
set -a && source ~/.envvars && set +a
export HF_HOME=/home/vllm/hf-cache
export VLLM_USE_FLASHINFER_SAMPLER=0
nohup vllm-env/bin/vllm serve google/gemma-4-E4B-it \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.93 \
  --max-model-len 8192 --max-num-seqs 8 \
  > logs/vllm-serve.log 2>&1 &
curl localhost:8000/v1/models
```

Startup takes about 3 minutes once the weights are cached (torch.compile + CUDA graphs).
Result: 118,535 tokens of KV cache, 14.5x concurrency at 8,192 tokens/request; about
11.2-11.4 GB of each 12 GB card in use.

For a single user at the model's full context window (131072 tokens), use
`--max-model-len 131072 --max-num-seqs 1` instead of `--max-model-len 8192 --max-num-seqs 8`.
That starts with the same `--gpu-memory-utilization 0.93` and reports 277,680 tokens of KV cache
(Gemma 4's sliding-window layers keep the cache small), about 11.5 + 11.3 GB used. A 127k-token
prompt completes in ~110 s (prefill) and decodes at ~40 tok/s.

For a coding agent (OpenCode etc.) add tool calling, or the server never returns `tool_calls`:
`--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4`
(optionally `--enable-prompt-tokens-details` to see cached prompt tokens in `usage`). Check it with
`./tool_check.py --url http://127.0.0.1:8000`. Prefix caching is on by default.

For a cold-start test add `--enable-sleep-mode` and `export VLLM_SERVER_DEV_MODE=1`; then
`curl -X POST localhost:8000/sleep` frees the GPU memory (weights go to host RAM, ~15 GB) and
`curl -X POST localhost:8000/wake_up` reloads in ~3.5 s. A full server start is ~166 s.

Why each flag:

- `--tensor-parallel-size 2`: bf16 Gemma 4 E4B is ~15 GB, which does not fit one 12 GB card.
  Each GPU holds ~7.5 GB of weights. PCIe-only peer access works.
- `--max-model-len 8192`: for the short-prompt benchmark. The full 131072 also fits, see above
  (use `--max-num-seqs 1`).
- `--host 127.0.0.1`: no authentication; use `--api-key` if you bind to `0.0.0.0`.

## Gemma 4 26B-A4B, 4-bit (for the OpenCode benchmark)

Model: [`cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4`](https://huggingface.co/cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4)
at revision `76373001e106330a6815f237ef00838c7fed9356` (not gated, Apache-2.0). It is a
compressed-tensors `pack-quantized` W4A16 build (int4 weights, group size 32, symmetric, 16-bit
activations) of `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized`, i.e. the same QAT checkpoint family
as Ollama's `gemma4:26b-a4b-it-qat`. 17.2 GB download. vLLM runs it with Marlin kernels on Ampere
(`MarlinLinearKernel`, `MarlinExperts`). The NVFP4 builds need Blackwell and FP8 (~27 GB) does not
fit 2x12 GB, so int4 is the option here.

```bash
cd /home/vllm
set -a && source ~/.envvars && set +a
export HF_HOME=/home/vllm/hf-cache VLLM_USE_FLASHINFER_SAMPLER=0
nohup vllm-env/bin/vllm serve cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4 \
  --revision 76373001e106330a6815f237ef00838c7fed9356 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.93 \
  --max-model-len 131072 --max-num-seqs 2 \
  --language-model-only \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --enable-prompt-tokens-details \
  > logs/vllm-serve-26b.log 2>&1 &
```

- Served model id: `cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4`; from the LAN `http://cv-ai:8000/v1`
  (192.168.1.4). No `--api-key`: anyone on the LAN can use it while it runs.
- Ready about 2 minutes after launch once downloaded. 8.43 GiB of weights per GPU, 138,498 tokens of
  KV cache (bf16), so the full 131072 window fits for one request (1.06x concurrency at full length;
  `--max-num-seqs 2` lets a short side request, e.g. an agent's title call, run alongside).
  About 11.4 + 11.2 GB in use.
- `--language-model-only` is what makes 131072 fit: without it the vision and audio encoders take
  ~1.1 GiB per GPU and startup fails with "1.83 GiB KV cache is needed ... available 0.66 GiB",
  estimated maximum model length 8768. OpenCode sends text only, so nothing is lost. Images are
  rejected by this server.
- `./tool_check.py --url http://127.0.0.1:8000`: all 4 checks pass.
- Quick speed check (`bench_llm.py --context-tokens 1000 16384 --runs 1 --no-think`): TTFT 0.52 s at
  1k tokens and 8.6 s at 15.4k; decode 96 and 89 tok/s.
- Stopping: kill the API server by PID and check that the `VLLM::Worker_TP0/1` processes exit too.
  Killing only the engine left both workers orphaned (parent 1), still holding 11 GB each, until
  they were killed by PID.

## Problems hit, in order

### 1. `ValueError: No available memory for the cache blocks` (`--gpu-memory-utilization 0.85`)

```
Available KV cache memory: -0.17 GiB
```

Since vLLM 0.21 CUDA-graph memory is counted in the budget, so 0.85 behaves like 0.75
(log: "equivalent to --gpu-memory-utilization=0.7487"). After ~7.5 GB of weights and
activations per GPU nothing is left for KV cache. Raise the setting.

### 2. `FileNotFoundError: 'ninja'` during warmup (at 0.95)

vLLM's FlashInfer top-k/top-p sampler JIT-compiles a kernel on first use, which needs
`ninja` and `nvcc`; this host has neither. Fix: `VLLM_USE_FLASHINFER_SAMPLER=0` (falls back to
the PyTorch sampler). Alternative: install the CUDA toolkit and `ninja`.

### 3. `torch.OutOfMemoryError` on GPU 0 (at 0.95)

Past the sampler, CUDA graph capture ran out of memory: GPU 0 has only 11.63 GiB usable and
Open WebUI (`uvicorn open_webui.main:app`, pid 3082) holds ~208 MB of it. 0.93 with
`--max-num-seqs 8` (fewer/smaller captured graphs) starts cleanly. If you stop
Open WebUI first, 0.95 may work.

### 4. Misleading "ready" detection

Compiler warnings from Triton's JIT contain the word `error`. Wait for
`Application startup complete.` in the log, not a loose grep for `error`.

### Stopping the server

`pkill -f "vllm serve"` run from a shell whose own command line contains that text kills the
shell. Use a bracketed pattern: `pkill -f "[v]llm-env/bin/vllm serve"`, then wait until
`nvidia-smi` shows the memory released before starting anything else on the GPUs.

## Smoke test

```bash
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"google/gemma-4-E4B-it","messages":[{"role":"user","content":"Say hi"}],"max_tokens":30}'
```

## Sharing the GPUs with Ollama

vLLM reserves ~93% of both cards while running, so it and an Ollama model cannot be loaded
together. Unload Ollama (`keep_alive: 0`, `/api/ps` empty) before starting vLLM, and stop vLLM
before loading an Ollama model. Benchmark 02 runs them in sequence; see
[benchmarks/02-dual-cuda-ollama-vs-vllm](benchmarks/02-dual-cuda-ollama-vs-vllm/).
