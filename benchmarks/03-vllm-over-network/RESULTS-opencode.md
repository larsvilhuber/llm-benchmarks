# OpenCode agent runs (benchmark 03, Part 2)

Real OpenCode sessions driven by `agent_bench.py` on the tasks in `tasks/`, judged by outcome.
See "Coding-agent workflow" in `TOOL.md` for how to run.

## Ollama on dual-cuda, Gemma 4 26B-A4B QAT, 128k context (2026-10-19)

| | |
|---|---|
| Server | Ollama on dual-cuda (2x RTX 3060 12 GB), one request at a time |
| Model | `gemma4:26b-a4b-it-qat-ctx128k`: `gemma4:26b-a4b-it-qat` (Q4_0, 25.2B total, about 4B active) with `num_ctx 131072` |
| Memory | 15.6 GB loaded, 15.5 GB on GPU (`/api/ps`) |
| Agent | OpenCode 1.19.3, `--context 131072`, thinking left at the model default |
| Runs | 3 per task, timeout 900 s |
| Raw data | `results-opencode-ollama-gemma4-26b-qat-128k.json` |

| Task | Pass | Median wall | Median steps | Median tool calls | Median tokens in / out |
|---|---|---|---|---|---|
| explain | 3/3 | 38 s | 4 | 3 | 43.6k / 368 |
| fix-bug | 3/3 | 46 s | 6 | 5 | 67.6k / 1.9k |
| implement | 3/3 | 48 s | 7 | 6 | 79.0k / 1.5k |
| refactor | 2/3 | 64 s | 11 | 10 | 125.3k / 1.4k |
| **all** | **11/12** | | | | |

"Tokens in" is summed over all steps of a run, including prompt-cache reads, so it counts the resent
history each step; it is not the context size of any one request.

Reading these:

- The 26B MoE model works as an OpenCode backend on dual-cuda: every task type passed, each run in about
  half a minute to a minute and a half.
- The one failure is genuine: in refactor run 1 the agent renamed the definition in `shop/pricing.py`
  but did not update its two callers, so the tests could not import.
- Compare the smoke test with Gemma 4 E4B bf16 on the same server (`results-opencode-ollama-smoke.json`,
  1 run per task): 1/4 passed, with a wrong answer, a give-up and a 400 s timeout. The larger model is the
  difference between unusable and usable here.
- Why 128k and not 256k: a `num_ctx 262144` tag of the same model (`gemma4:26b-a4b-it-qat-ctx256k`, kept on
  dual-cuda) loads at 16.0 GB with only 13.2 GB on GPU; the remainder runs on the CPU and would slow every step.
  These tasks need far less than 128k.
- Caveats: 3 runs per task is a small sample; the tasks are small (one to three files); dual-cuda may have been
  shared with other work during the run (not checked).

## Three backends, same model family: Ollama vs vLLM on dual-cuda vs vLLM on mac-studio (2026-10-25)

Gemma 4 26B-A4B QAT at 4-bit on every backend, each in its engine's own format, 131072 context, the
same tasks, 3 runs per task, timeout 900 s, one backend at a time (sequential, nothing else on the servers).

| | Ollama, dual-cuda | vLLM, dual-cuda | vLLM, mac-studio |
|---|---|---|---|
| Model | `gemma4:26b-a4b-it-qat-ctx128k` (Q4_0 GGUF) | `cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4` @ `76373001` (compressed-tensors W4A16) | `mlx-community/gemma-4-26B-A4B-it-qat-4bit` (MLX 4-bit) |
| Engine | Ollama (llama.cpp runner) | vLLM 0.31.0, CUDA, TP=2 | vllm-metal 0.31.0 |
| Server flags | `num_ctx 131072` | see `README-cuda.md` (`--language-model-only`, tool and reasoning parsers `gemma4`, `--enable-prompt-tokens-details`) | tool and reasoning parsers `gemma4`; no sleep mode (not supported on Metal) |
| Access | LAN | LAN | SSH tunnel |
| `tool_check.py` | (not run on this model) | 4/4 | 4/4 |
| Raw data | `results-opencode-ollama-gemma4-26b-qat-128k.json` | `results-opencode-vllm-cv-gemma4-26b-qat-128k.json` | `results-opencode-vllm-mac-gemma4-26b-qat-128k.json` |

Pass count and median wall time of the **passing** runs:

| Task | Ollama, dual-cuda | vLLM, dual-cuda | vLLM, mac-studio |
|---|---|---|---|
| explain | 3/3, 38 s | 3/3, 11 s | 3/3, 37 s |
| fix-bug | 3/3, 46 s | 2/3, 20 s | 2/3, 38 s |
| implement | 3/3, 48 s | 1/3, 19 s | 1/3, 55 s |
| refactor | 2/3, 78 s | 2/3, 21 s | 2/3, 44 s |
| **all** | **11/12**, 46 s | **8/12**, 16 s | **8/12**, 42 s |

Reading these:

- **Speed:** when a run succeeds, vLLM on dual-cuda is about 3x faster than Ollama on the same GPUs, and
  mac-studio is about as fast as Ollama on dual-cuda. vLLM on dual-cuda reported 98% of prompt tokens served from
  its prefix cache across the run.
- **Reliability:** both vLLM servers passed 8/12, Ollama 11/12, and both vLLM servers failed in the same two
  ways, which Ollama never did:
  1. **Empty turn** (3 runs on each vLLM server): after reading the file, the model ends its turn with
     about 3 output tokens, no text and no edit, so OpenCode stops. On dual-cuda, implement runs 0 and 2 were
     identical to the token (33,162 in / 79 out), so this is deterministic for a given context.
  2. **Edit loop** (1 run on each): the model copies the line-number prefixes of OpenCode's read output
     (`7:     text = ...`) into the edit's search string, every edit fails with "Could not find oldString",
     and it repeats nearly the same edit until the timeout (153 failed edits on mac-studio, 238 on dual-cuda).
- Two vLLM servers with different quantizations and kernels (CUDA AWQ vs Metal MLX) fail identically while
  Ollama does not, so the cause is more likely on the vLLM side (its Gemma 4 chat template, the `gemma4`
  reasoning or tool parser, or default sampling settings) than in the 4-bit weights. **Not tested.**
- mac-studio's vLLM reports 0 cached prompt tokens on every step. That is because it was not started with
  `--enable-prompt-tokens-details` (which turns on the reporting); it does not show whether caching works there.
- Caveats: 3 runs per task; small tasks. The two vLLM servers were checked idle by their agents before the runs.

Next: find the vLLM failure cause (compare the prompt OpenCode sends as rendered by each server's chat
template; try without `--reasoning-parser`; check vLLM's default temperature against Ollama's), then rerun.
