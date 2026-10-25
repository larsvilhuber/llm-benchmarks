# OpenCode agent runs (benchmark 03, Part 2)

Real OpenCode sessions driven by `agent_bench.py` on the tasks in `tasks/`, judged by outcome.
See "Coding-agent workflow" in `TOOL.md` for how to run.

## Ollama on cv-ai, Gemma 4 26B-A4B QAT, 128k context (2026-10-19)

| | |
|---|---|
| Server | Ollama on cv-ai (2x RTX 3060 12 GB), one request at a time |
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

- The 26B MoE model works as an OpenCode backend on cv-ai: every task type passed, each run in about
  half a minute to a minute and a half.
- The one failure is genuine: in refactor run 1 the agent renamed the definition in `shop/pricing.py`
  but did not update its two callers, so the tests could not import.
- Compare the smoke test with Gemma 4 E4B bf16 on the same server (`results-opencode-ollama-smoke.json`,
  1 run per task): 1/4 passed, with a wrong answer, a give-up and a 400 s timeout. The larger model is the
  difference between unusable and usable here.
- Why 128k and not 256k: a `num_ctx 262144` tag of the same model (`gemma4:26b-a4b-it-qat-ctx256k`, kept on
  cv-ai) loads at 16.0 GB with only 13.2 GB on GPU; the remainder runs on the CPU and would slow every step.
  These tasks need far less than 128k.
- Caveats: 3 runs per task is a small sample; the tasks are small (one to three files); cv-ai may have been
  shared with other work during the run (not checked).
