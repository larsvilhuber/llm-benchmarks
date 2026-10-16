# 02: Ollama vs vLLM on the same hardware

Both engines run on the same host (cv-ai, 2x12 GB CUDA), one after the other, so the
difference is the engine and not the machine.

## Prerequisite (separate task)

Install vLLM on cv-ai in `/home/vllm` (own virtualenv) and
confirm `vllm serve` starts with the target model.

## Do the servers conflict?

Both want the GPUs, but you do not need to stop the Ollama container as long as
Ollama has **unloaded the model**:

- Ollama frees the model's VRAM when it unloads (default after 5 minutes idle).
  Force it with `curl $OLLAMA_HOST/api/generate -d '{"model":"<m>","keep_alive":0}'`
  and confirm `curl $OLLAMA_HOST/api/ps` lists no models.
  An idle container still holds a small CUDA context (a few hundred MB per GPU).
- vLLM preallocates `--gpu-memory-utilization` (default 0.9) of each GPU at
  startup and **refuses to start if that much is not free**. Check `nvidia-smi`
  before launching.
- For the cleanest numbers, stop the Ollama container while vLLM runs and start it
  again for the Ollama pass, so neither has resident state. Run them in sequence,
  never together.
- Do not load an Ollama model while vLLM is running; vLLM keeps its reservation
  and Ollama would fall back to CPU or fail.

## Fitting the model

bf16 Gemma 4 E4B is about 15 GB, which does not fit one 12 GB card. Ollama
splits layers across both GPUs automatically. vLLM needs
`--tensor-parallel-size 2` (needs NCCL-capable peer access; PCIe-only is fine
but slower) or a quantized/fp8 build. Whichever is used must be recorded in the
results, because it changes what is being compared.

## Procedure

1. Ollama pass: container up, model unloaded, vLLM not running.
   `./bench_llm.py --cold-warm ...` for the Ollama side only (see the note below).
2. Stop Ollama (or confirm empty `/api/ps` and free VRAM), start vLLM, wait for ready.
3. vLLM pass with the same prompts, `--max-tokens` and runs.
4. Record `nvidia-smi` memory use during each pass.

Note: `bench_llm.py` currently drives both endpoints in one invocation. Running them
in sequence needs a single-engine mode (`--only ollama|vllm`), to be added with this
benchmark.

## Results

Run 2026-10-16 on cv-ai, one engine at a time (the other stopped / unloaded, GPUs free
before each pass). Same prompts, `--max-tokens 256`, 3 runs (median), `temperature=0`.

| | Ollama | vLLM 0.31.0 |
|---|---|---|
| Model | `gemma4:e4b-it-bf16` (GGUF, llama.cpp runner) | `google/gemma-4-E4B-it` (bf16 safetensors) |
| GPU split | layers across both GPUs (automatic) | `--tensor-parallel-size 2` |
| Context | 32768 (reported by `/api/ps`) | `--max-model-len 8192` |
| GPU memory in use | 6.3 GB + 5.5 GB | 11.4 GB + 11.2 GB (`--gpu-memory-utilization 0.93`) |
| Launch | default Ollama container | see [README-cuda.md](../../README-cuda.md) |

| Prompt | Ollama TTFT | vLLM TTFT | Ollama tok/s | vLLM tok/s |
|---|---|---|---|---|
| short | 165 ms | 47 ms | 33.2 | 57.5 |
| medium | 180 ms | 42 ms | 33.5 | 56.5 |
| long | 179 ms | 30 ms | 33.4 | 56.5 |
| long x4 parallel (aggregate) | 11,890 ms | 79 ms | 32.7 | 180.1 |

Raw data: `results-ollama.json`, `results-vllm.json`.

Reading these:

- Decode is about 1.7x faster on vLLM (56-57 vs 33 tok/s) and TTFT 4-6x lower.
- Under 4 concurrent requests vLLM batches (180 tok/s aggregate); Ollama's aggregate stays at
  its single-stream rate, i.e. requests were served one after another (its
  `OLLAMA_NUM_PARALLEL` was not checked; this is the observed behaviour with the container's
  current settings).
- **Output is not the same.** For the short prompt Ollama emits a thinking trace
  (`reasoning` filled, `content` empty, `finish_reason: length` at 256 tokens) while vLLM
  answers directly in 57 tokens. Total times and token counts are therefore not
  comparable; TTFT and tok/s are. Enabling or disabling thinking identically on both would
  fix this and is a follow-up.
- Not like-for-like in other ways: different runners (llama.cpp vs vLLM kernels), context
  length 32768 vs 8192, and the vLLM run needed `VLLM_USE_FLASHINFER_SAMPLER=0` (PyTorch
  sampler instead of FlashInfer). Cold start (`--cold-warm`) was not run: vLLM was not started
  with sleep mode.
