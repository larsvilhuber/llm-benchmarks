# 03: Large models and coding-agent backends, cv-ai vs mac-studio

Status: the OpenCode runs of Part 2 are done ([RESULTS-opencode.md](RESULTS-opencode.md)).
Part 1 (capacity) and Claude Code as the agent are still to do. This page is the original plan.

## Questions

1. **Capacity:** what is the largest model, and the longest context, each host can serve
   usefully? cv-ai has 2x12 GB of VRAM (CUDA); mac-studio has about 50 GB of unified memory
   (Metal). Each wins on a different axis (CUDA throughput vs memory size), so which wins
   depends on model size.
2. **Agent backend:** can either host act as the private backend for a coding agent? Scope
   for now is Claude Code pointed at the host, and the focus is metrics (Part 1 and the
   measurable parts of Part 2). OpenCode, first planned for a later round, was run first;
   Claude Code is still to do.

Clients reach the hosts directly over the LAN from a workstation, which is the real
usage. Network overhead is not isolated; record round-trip time (`ping`) once per run.
Security and concurrency are out of scope.

## Part 1: capacity

For each candidate model on each host, record: loads or not, load time, memory used (weights,
KV cache), and the largest `--max-model-len` that starts reliably. Then, at that context:

- **Context sweep** (new `bench_llm.py` option): prompts of about 2k, 8k, 16k, 32k, 64k tokens
  (as far as the host goes). Report prefill time (TTFT), decode tok/s, and memory, and TTFT
  again on an immediate repeat of the same prompt (prefix-cache hit).
- Short-prompt TTFT and decode tok/s, as in benchmarks 01 and 02, for continuity.

Candidates are chosen to bracket each host's limit. Starting points, already present in
cv-ai's Ollama (so known to exist; vLLM needs a compatible format for each host):

| Model (class) | Why |
|---|---|
| Gemma 4 E4B (about 8B) | baseline from 01/02 |
| gpt-oss 20B | fits both; tool-calling oriented |
| Gemma 4 26B-A4B (MoE) | mid size, few active parameters |
| Qwen3-Coder 30B (MoE) | coding-specific; about 18 GB at 4-bit, the likely cv-ai limit |
| Qwen 27B-class dense | dense models stress memory bandwidth |
| a 70B-class model at 4-bit (about 40 GB) | probably only mac-studio can hold it |

Quantization will differ per host (AWQ/GPTQ/FP8 on CUDA, MLX 4/8-bit on Metal), so each
result states the exact model repo and quantization; "same model" means same weights family,
not byte-identical files. Expectation to test, not assume: MoE models with few active parameters
suit both hosts, while dense models above about 24 GB cannot run on cv-ai at all.

## Part 2: coding-agent backend

### Compatibility checks (do first; each is a pass/fail)

| Check | How |
|---|---|
| OpenAI-compatible chat + streaming works | already used by `bench_llm.py` |
| Tool calling works | start vLLM with `--enable-auto-tool-choice --tool-call-parser <parser for the model>`; send a request with a tool schema and check for a well-formed call |
| Anthropic Messages API (`/v1/messages`) | needed by Claude Code; confirm the installed vLLM version serves it (otherwise a translating proxy such as LiteLLM is required, which then belongs in the setup notes) |
| Context large enough for the agent | Claude Code's system prompt plus tool definitions alone are many thousands of tokens; sessions grow into the tens of thousands. Need a usable context of at least 32k, ideally 64k+ |
| Prefix caching on | agents resend a growing identical prefix every turn; without caching each step re-prefills everything |

Claude Code is pointed at a host with environment variables only (no code changes):
`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` (any placeholder the server accepts), and
`ANTHROPIC_MODEL` (plus the small/fast model variable, set to the same model). Open agents are
pointed at the OpenAI-compatible base URL in their config (deferred). The exact settings
used are recorded in this directory for each agent and host.

### Task suite

A small fixed set of tasks in a pinned repository with an automatic pass/fail check
(its test command), so a run is judged by an outcome, not by reading output:

1. fix a failing unit test (one-file bug)
2. implement a function from a docstring and make the tests pass
3. a small refactor touching several files (rename + update callers), tests must still pass
4. a "find and explain" task over the repo (answer checked against a known key)

Each task runs in a fresh copy of the repository. Each (agent, host, model, task) is run
several times (sampling is not deterministic; start with 5) and reported as pass rate, with
median wall time, number of tool calls, and tokens in/out. A new `agent_bench.py` drives
the agents headlessly (for Claude Code, `claude -p`), collects these numbers, and writes
`results-agent-*.json`. It lives beside `bench_llm.py` and reads host URLs from the
environment like the rest.

Reported alongside: time spent in model inference vs tools, and the longest context reached.
A reference run on a hosted model is optional, to calibrate whether a failure is the task
or the local model.

## Tool changes needed (after 02 merges, to avoid editing `bench_llm.py` concurrently)

1. Generalize the fixed targets (vllm, ollama) to named targets,
   `--target NAME=URL[,model=M][,tunnel]`, so any endpoints can be compared. The `--only`
   flag from 02 becomes target selection.
2. Context-length sweep with deterministic filler prompts of N tokens, and a repeat request
   to measure prefix-cache hits.
3. A tool-call check (send a tool schema, validate the response).
4. New `agent_bench.py` and a `tasks/` directory with the pinned repo and checks. Written
   with a pluggable agent command so adding opencode later is a small change.

## Procedure

1. Confirm 02 is done and merged; record vLLM version and flags for both hosts in `STATUS.md`.
2. Run the compatibility checks on each host with the smallest model that passes tool
   calling, and note which hosts pass.
3. Part 1: capacity sweep per model per host. Stop Ollama on cv-ai while vLLM runs (GPU
   memory; see 02).
4. Part 2: for each host and the best model that fits it, run the task suite with Claude Code.
   Open agents (opencode first) are a later round.
5. Store raw results here; fill in the tables and a short conclusion on what each host is
   good for.

## Results

- Part 2, OpenCode on three backends: [RESULTS-opencode.md](RESULTS-opencode.md); summary in
  [FINDINGS.md](../../FINDINGS.md).
- Part 1 and Claude Code: not run yet.
