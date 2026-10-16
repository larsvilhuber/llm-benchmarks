# 03: vLLM over the network, cv-ai vs mac-studio (planned)

**Blocked until benchmark 02 finishes.** It reuses 02's vLLM install and launch
configuration on cv-ai, and 02's vLLM-on-cv-ai results serve as the local baseline.

## Question

How do the two vLLM deployments compare when a client reaches them over the network, as
real users would: vLLM on cv-ai (Linux, CUDA, 2x12 GB) vs vLLM on mac-studio (Apple
silicon, vllm-metal)? And how much of the observed latency is the network?

Same model on both (`google/gemma-4-E4B-it`, bf16). Benchmark 01 compared Ollama on cv-ai
with vLLM on mac-studio; 02 compares the engines on cv-ai; 03 compares the two vLLM
hosts to each other.

## Setup

| | cv-ai | mac-studio |
|---|---|---|
| Engine | vLLM (CUDA, config from 02) | vLLM (vllm-metal, see `README-mlx.md`) |
| Access | direct, firewall opened to the client | direct if reachable, else SSH tunnel (`--ssh-tunnel`) |

- **Firewall/auth (cv-ai):** open port 8000 only to the client's address, not the whole
  network. A vLLM server bound to a network interface should require a key
  (`vllm serve ... --api-key`); the tool must send it (see "Tool changes").
- **Stop Ollama while vLLM runs on cv-ai** (GPU memory; see 02).

## Client placement (the comparison that isolates the network)

| Run | Client on | Purpose |
|---|---|---|
| A | the server host itself (loopback) | engine-only baseline; for cv-ai this is 02's vLLM result |
| B | a third machine on the LAN | what a user sees, per host |

Overhead from the network = B minus A, per host. Run the clients one host at a time, never
both servers at once from the same client, so client CPU and NIC are not shared.
Before each B run, record round-trip time (`ping`) and, if available, `iperf3` bandwidth
between client and server.

## Measurements

- TTFT and decode tok/s at concurrency 1, 2, 4, 8, 16. Throughput scaling is where the
  two hosts should differ most (continuous batching on CUDA vs Metal).
- Prompt length: the existing short/medium/long, plus prefill-heavy prompts (about 2k and 8k
  tokens) to expose prefill speed and request-size effects on the network.
- Cold/warm as in 01 (`--cold-warm`), if sleep mode is enabled on the host.
- Resource use during each pass: `nvidia-smi` on cv-ai, memory pressure on mac-studio.

## Tool changes needed (after 02 merges, to avoid editing `bench_llm.py` concurrently)

1. Generalize the two fixed targets (vllm, ollama) to named targets, e.g.
   `--target NAME=URL[,model=M][,tunnel]`, so any two or more endpoints can be compared.
   Keep `--vllm-url/--ollama-url` working. The `--only` flag from 02 becomes `--target` selection.
2. `--api-key` (or `VLLM_API_KEY` from the environment) sent as a bearer token.
3. Concurrency sweep (`--concurrency 1 2 4 8 16`) with per-level aggregate throughput.
4. A prompt-length option to generate prefill-heavy prompts of N tokens deterministically.

## Procedure

1. Confirm 02 is done and merged; record vLLM version and flags for both hosts in `STATUS.md`.
2. Open the firewall on cv-ai for the client; start vLLM with an API key; check `curl` from the client.
3. Run A on each host, then B for each host, with identical arguments.
4. Store raw results as `results-<host>-<A|B>.json` in this directory; fill in the results tables.

## Results

Not run yet.
