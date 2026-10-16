#!/usr/bin/env python3
"""Compare latency/throughput of two OpenAI-compatible LLM endpoints (e.g. vLLM vs Ollama).

Both servers are driven through the same /v1/chat/completions streaming API so the
measurements are like-for-like. Endpoints come from arguments or environment:

  VLLM_URL    base URL of the vLLM server   (e.g. http://host:8000)
  OLLAMA_URL  base URL of the Ollama server (else derived from OLLAMA_HOST)

Example:
  ./bench_llm.py --vllm-url "$VLLM_URL" --ollama-model gemma4:e4b --wait 3600
"""
import argparse
import atexit
import json
import os
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

PROMPTS = {
    "short": "Explain in two sentences what a hash table is.",
    "medium": "Write a Python function that merges overlapping intervals, with a brief explanation "
              "of its time complexity and two test cases.",
    "long": "Write a detailed essay (about 500 words) on how the transformer architecture changed "
            "natural language processing, covering attention, scaling, and pretraining.",
}


def normalize(url):
    if not url:
        return url
    if "://" not in url:
        url = "http://" + url
    return url.rstrip("/")


def open_tunnel(url):
    """Forward a free local port to the vLLM host over SSH; return the local base URL."""
    u = urlparse(url)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        local_port = s.getsockname()[1]
    proc = subprocess.Popen(
        ["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "BatchMode=yes",
         "-L", f"127.0.0.1:{local_port}:127.0.0.1:{u.port}", u.hostname],
        stderr=subprocess.PIPE)
    atexit.register(proc.terminate)
    for _ in range(50):
        if proc.poll() is not None:
            sys.exit(f"ssh tunnel to {u.hostname} failed: {proc.stderr.read().decode().strip()}")
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", local_port)) == 0:
                return f"http://127.0.0.1:{local_port}"
        time.sleep(0.2)
    sys.exit("ssh tunnel did not come up")


def http_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def wait_ready(base, deadline, label):
    """Poll /v1/models until the server answers or the deadline passes."""
    while True:
        try:
            return http_json(base + "/v1/models", timeout=5)
        except (urllib.error.URLError, OSError, ValueError) as e:
            if time.time() >= deadline:
                sys.exit(f"{label}: not ready at {base} ({e})")
            print(f"{label}: waiting for {base} ...", file=sys.stderr)
            time.sleep(15)


def stream_chat(base, model, prompt, max_tokens, timeout=600):
    """One streaming request. Returns dict of timings and token counts."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    chunks = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                d = ch.get("delta", {})
                # thinking models may emit reasoning separately from content
                if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                    chunks += 1
                    if first is None:
                        first = time.perf_counter()
    t1 = time.perf_counter()
    if first is None:
        first = t1
    out_tokens = usage.get("completion_tokens") or chunks
    decode_time = t1 - first
    return {
        "ttft": first - t0,
        "total": t1 - t0,
        "prompt_tokens": usage.get("prompt_tokens"),
        "out_tokens": out_tokens,
        "tps": out_tokens / decode_time if decode_time > 0 and out_tokens > 1 else float("nan"),
    }


def post(url, body=None, timeout=600):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def ollama_unload(base, model):
    post(base + "/api/generate", {"model": model, "keep_alive": 0})
    for _ in range(30):
        if not any(m["name"] == model for m in http_json(base + "/api/ps").get("models", [])):
            return True
        time.sleep(1)
    return False


def vllm_supports_sleep(base):
    try:
        http_json(base + "/is_sleeping")
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def vllm_sleep(base):
    post(base + "/sleep?level=1")


def vllm_wake(base):
    t0 = time.perf_counter()
    post(base + "/wake_up")
    return time.perf_counter() - t0


def bench(label, base, model, args, warmup=True):
    print(f"\n=== {label}: {base}  model={model} ===")
    if warmup:
        print("warmup ...", file=sys.stderr)
        stream_chat(base, model, PROMPTS["short"], 16)
    results = {}
    for name in args.prompts:
        runs = [stream_chat(base, model, PROMPTS[name], args.max_tokens) for _ in range(args.runs)]
        results[name] = summarize(name, runs, "sequential")
    if args.concurrency > 1:
        name = args.prompts[-1]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(args.concurrency) as ex:
            runs = list(ex.map(lambda _: stream_chat(base, model, PROMPTS[name], args.max_tokens),
                               range(args.concurrency)))
        wall = time.perf_counter() - t0
        agg = sum(r["out_tokens"] for r in runs) / wall
        results[f"{name}@c{args.concurrency}"] = summarize(
            f"{name} x{args.concurrency} parallel", runs, f"aggregate {agg:.1f} tok/s")
        results[f"{name}@c{args.concurrency}"]["agg_tps"] = agg
    return results


def summarize(name, runs, note):
    med = lambda k: statistics.median(r[k] for r in runs)
    s = {"first_ttft": runs[0]["ttft"], "ttft": med("ttft"), "tps": med("tps"), "total": med("total"), "out_tokens": med("out_tokens")}
    print(f"{name:<22} TTFT {s['ttft']*1000:7.0f} ms | {s['tps']:7.1f} tok/s | "
          f"total {s['total']:6.2f} s | {s['out_tokens']:.0f} tok  ({note}; 1st-req TTFT {s['first_ttft']*1000:.0f} ms)")
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vllm-url", default=os.environ.get("VLLM_URL"))
    p.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL") or os.environ.get("OLLAMA_HOST"))
    p.add_argument("--vllm-model", help="default: first model reported by vLLM /v1/models")
    p.add_argument("--ollama-model", required=True)
    p.add_argument("--prompts", nargs="+", choices=PROMPTS, default=list(PROMPTS))
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--concurrency", type=int, default=4, help="parallel requests test; 1 disables")
    p.add_argument("--ssh-tunnel", action="store_true",
                   help="reach vLLM through an SSH tunnel to the host in --vllm-url")
    p.add_argument("--cold-warm", action="store_true",
                   help="run a cold pass (model unloaded first), then a warm pass, then unload "
                        "(Ollama: keep_alive 0; vLLM: /sleep and /wake_up if the server has sleep mode)")
    p.add_argument("--only", choices=["ollama", "vllm"],
                   help="benchmark one engine only (run the engines in sequence on shared GPUs)")
    p.add_argument("--wait", type=int, default=0, help="seconds to wait for servers to come up")
    p.add_argument("--json", help="write results to this file")
    args = p.parse_args()

    vllm, ollama = normalize(args.vllm_url), normalize(args.ollama_url)
    run_vllm, run_ollama = args.only != "ollama", args.only != "vllm"
    if (run_vllm and not vllm) or (run_ollama and not ollama):
        p.error("need --vllm-url/VLLM_URL and --ollama-url/OLLAMA_URL/OLLAMA_HOST "
                "(only the one selected by --only)")

    if run_vllm and args.ssh_tunnel:
        vllm = open_tunnel(vllm)
    deadline = time.time() + args.wait
    if run_vllm:
        models = wait_ready(vllm, deadline, "vllm")
        vllm_model = args.vllm_model or models["data"][0]["id"]
    if run_ollama:
        wait_ready(ollama, deadline, "ollama")

    if not args.cold_warm:
        out = {}
        if run_vllm:
            out["vllm"] = {"warm": bench("vLLM", vllm, vllm_model, args)}
        if run_ollama:
            out["ollama"] = {"warm": bench("Ollama", ollama, args.ollama_model, args)}
    else:
        out = {}
        if run_ollama:
            out["ollama"] = {}
            # Ollama: unload, cold pass loads the model, warm pass reuses it, then unload.
            if not ollama_unload(ollama, args.ollama_model):
                print("ollama: model still loaded after unload request", file=sys.stderr)
            out["ollama"]["cold"] = bench("Ollama (cold)", ollama, args.ollama_model, args, warmup=False)
            out["ollama"]["warm"] = bench("Ollama (warm)", ollama, args.ollama_model, args, warmup=False)
            ollama_unload(ollama, args.ollama_model)
            print("ollama: model unloaded", file=sys.stderr)
        if run_vllm:
            out["vllm"] = {}
            # vLLM: only a server started with sleep mode can release the model.
            if vllm_supports_sleep(vllm):
                vllm_sleep(vllm)
                print(f"vllm: wake_up (reload) took {vllm_wake(vllm):.2f} s", file=sys.stderr)
            else:
                print("vllm: sleep mode not enabled on server; 'cold' is only the first pass on the "
                      "running server (no weight reload). Start vLLM with --enable-sleep-mode and "
                      "VLLM_SERVER_DEV_MODE=1 for a true cold start.", file=sys.stderr)
            out["vllm"]["cold"] = bench("vLLM (cold)", vllm, vllm_model, args, warmup=False)
            out["vllm"]["warm"] = bench("vLLM (warm)", vllm, vllm_model, args, warmup=False)

    if "vllm" in out and "ollama" in out:
        for phase in out["ollama"]:
            print(f"\n=== speedup, {phase} (vLLM vs Ollama) ===")
            for k in out["vllm"][phase]:
                v, o = out["vllm"][phase][k], out["ollama"][phase][k]
                print(f"{k:<22} decode {v['tps']/o['tps']:.2f}x | TTFT {o['ttft']/v['ttft']:.2f}x faster")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
