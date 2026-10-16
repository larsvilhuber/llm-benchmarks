# 02: Ollama vs vLLM on the same hardware (planned)

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

Not run yet.
