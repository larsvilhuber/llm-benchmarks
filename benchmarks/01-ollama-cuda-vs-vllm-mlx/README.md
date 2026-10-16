# 01: Ollama + CUDA vs vLLM + MLX

| | Ollama | vLLM |
|---|---|---|
| Hardware | 2x 12 GB NVIDIA GPUs (CUDA) | Apple silicon, ~50 GB unified memory (MLX) |
| Model | `gemma4:e4b-it-bf16` (llama.cpp, BF16, 7.5B) | `google/gemma-4-E4B-it` (full precision) |
| Access | direct over the network | through an SSH tunnel |

Different hardware on each side: this compares deployed setups, not engines.

## Reproduce

```
export VLLM_URL=http://<vllm-host>:8000     # OLLAMA_HOST must also be set
# warm only
../../bench_llm.py --ssh-tunnel --ollama-model gemma4:e4b-it-bf16 --json results-warm.json
# cold then warm, with unload
../../bench_llm.py --ssh-tunnel --cold-warm --ollama-model gemma4:e4b-it-bf16 --json results-cold-warm.json
```

Ollama needs the matching weights first: `ollama pull gemma4:e4b-it-bf16`. The
default `gemma4:e4b` is 4-bit and is not comparable to vLLM's full precision.

## Results: warm (`results-warm.json`)

| Prompt | vLLM TTFT | vLLM tok/s | Ollama TTFT | Ollama tok/s |
|---|---|---|---|---|
| short | 51 ms | 45.1 | 164 ms | 33.5 |
| medium | 69 ms | 44.1 | 171 ms | 33.5 |
| long | 54 ms | 44.0 | 173 ms | 33.5 |
| long x4 parallel | 118 ms | 165 aggregate | 11.9 s | 32.7 aggregate |

vLLM decodes about 1.3x faster and streams its first token about 3x sooner. With four
parallel requests vLLM batches them (about 165 tok/s aggregate); Ollama ran them one
at a time, probably because its default is one parallel request (`OLLAMA_NUM_PARALLEL`).

## Caveats

- On the short prompt vLLM produced 59 tokens and Ollama 256 (on every prompt).
  Ollama may be emitting a thinking trace or not stopping at end of reply. Not
  investigated; it affects totals, not tok/s or TTFT.
- vLLM was not restarted or put to sleep (sleep mode is not enabled on that server), so its
  "cold" pass is only the first pass on a running server, not a weight reload.

## Results: cold vs warm (`results-cold-warm.json`)

Time to first token of the **first request** of each pass (the median over 3 runs hides
a one-time load):

| | Cold | Warm |
|---|---|---|
| Ollama | 6410 ms (model load) | 171 ms |
| vLLM | 553 ms (first request, no reload) | 50 ms |

Steady-state numbers are the same in both passes (Ollama about 33 tok/s, vLLM about 45 tok/s).
Ollama unloads the model after the cold/warm run (`keep_alive: 0`).
