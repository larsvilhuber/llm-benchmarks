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

## Results: maximum context, single user

Same model length on both sides: 131072 tokens (the model's maximum), one request at a time.

- vLLM: `--max-model-len 131072 --max-num-seqs 1 --tensor-parallel-size 2 --gpu-memory-utilization 0.93`
  (277,680 tokens of KV cache, so the full window fits with room to spare).
- Ollama: tag `gemma4:e4b-it-bf16-ctx128k`, created from `gemma4:e4b-it-bf16` with
  `PARAMETER num_ctx 131072` (`/api/create`); `/api/ps` confirms `context_length: 131072`.
- Prompts: a filler word list of about N tokens plus a short question, unique per request
  (random id at the start) so neither engine's prefix cache can hit. `--max-tokens 128`,
  `temperature=0`, 2 runs (median), `bench_llm.py --context-tokens ...`. "In" is the prompt size
  each server counted (tokenizers differ by a few tokens).

| Prompt tokens in (vLLM / Ollama) | TTFT vLLM | TTFT Ollama | Decode vLLM | Decode Ollama |
|---|---|---|---|---|
| 3,884 / 3,900 | 1.8 s | 1.6 s | 54.5 tok/s | 33.3 tok/s |
| 15,398 / 15,391 | 7.7 s | 4.9 s | 50.7 tok/s | 32.3 tok/s |
| 61,422 / 61,556 | 39.8 s | 23.0 s | 45.7 tok/s | 28.6 tok/s |
| 127,622 / 127,620 | 110.5 s | 63.2 s | 40.0 tok/s | 24.6 tok/s |

Raw data: `results-ctx-vllm.json`, `results-ctx-ollama.json`.

GPU memory: vLLM 11.5 + 11.3 GB (preallocated, constant). Ollama's loaded model is 11.0 GB in
total (`/api/ps`); sampled `nvidia-smi` peaks were 7.9 GB (GPU 0) and 6.3 GB (GPU 1), sampled every
20 s so they may undershoot.

Reading these:

- Both engines hold the full 131072-token window on 2x12 GB. Nothing needed quantizing.
- **Prefill (TTFT) favours Ollama once the prompt is long**: 1.7x faster at 61k-128k tokens,
  reversing the short-prompt result in the table above. Prefill cost grows roughly linearly on
  both (about 0.9 ms/token on vLLM, 0.5 ms/token on Ollama).
- **Decode favours vLLM** at every size (1.6x at 128k), and both slow down as the context grows
  (vLLM 54.5 to 40.0 tok/s, Ollama 33.3 to 24.6).
- Not investigated: why vLLM prefills slower here. Candidates are the attention backend (the log
  says FlashAttention 4 is unavailable for Gemma 4's mixed head dimensions, so `TRITON_ATTN` is
  used), tensor-parallel communication over PCIe, and the default `--max-num-batched-tokens`.
  Tuning those could change this result.
- Caveats as above: Ollama's output includes a thinking trace so output lengths differ (128 capped
  vs 57); decode tok/s is still comparable. vLLM used the PyTorch sampler. Two runs per size only.

## Results: cold start (131072 context, single user)

`bench_llm.py --cold-warm --only <engine> --context-tokens 4096 65536 --runs 2 --max-tokens 128`.
Same configuration as the max-context run; vLLM additionally started with
`VLLM_SERVER_DEV_MODE=1 ... --enable-sleep-mode`. The cold pass is the first request after the
model was released; the warm pass repeats it with the model resident.

| | vLLM | Ollama |
|---|---|---|
| How the model was released | `/sleep` (weights offloaded to host RAM, KV cache freed) | `keep_alive: 0` (unloaded, `/api/ps` empty) |
| Reload | `/wake_up` took 3.5 s | included in first request |
| First request, 3.9k tokens: TTFT cold | 2.8 s | 12.4 s |
| First request, 3.9k tokens: TTFT warm | 1.8 s | 2.3 s |
| Cold extra over warm | ~1.0 s (+3.5 s wake_up) | ~10.2 s |
| 61k-token request, TTFT cold / warm | 40.3 s / 40.1 s | 22.4 s / 22.6 s |
| Decode cold / warm (3.9k) | 59.6 / 54.6 tok/s | 33.2 / 33.2 tok/s |

Process start to `Application startup complete` for `vllm serve` (131072 context, TP=2,
weights already in the OS page cache, torch.compile cache already populated): **166 s**.

Raw data: `results-cold-warm-vllm.json`, `results-cold-warm-ollama.json`.

Reading these:

- Reloading is much cheaper on vLLM when it uses sleep mode (3.5 s plus ~1 s of first-request
  warmup), but only because the weights stay in host RAM. A real restart of the server is 166 s,
  versus roughly 10 s for Ollama to load the model on first use.
- After the first request the pattern from the max-context run holds: Ollama prefills the 61k
  prompt about 1.8x faster, vLLM decodes about 1.6x faster.
- Caveats: this is not disk-cold. Both engines read weights from the OS page cache (dropping caches
  needs root and was not done). Sleep mode keeps ~15 GB of weights in host RAM, so it is not free.
  The 166 s start time is a single measurement.
