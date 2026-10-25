#!/usr/bin/env python3
"""Run a coding agent (OpenCode) against one or more LLM servers on a fixed set of tasks.

Each (target, task, run) gets a fresh copy of the task's repository, its own git repo and
OpenCode config pointing at the target. The agent runs headless; the outcome is judged by
the task's check (tests pass, forbidden names gone, answer contains the key facts), not by
reading its output. Reports pass rate, wall time, agent steps, tool calls and tokens.

Targets are OpenAI-compatible servers (vLLM, Ollama, ...):

  --target NAME=URL[,model=M]     URL is the server base (http://host:port); /v1 is added.

Examples:
  ./agent_bench.py --target ollama=http://$OLLAMA_HOST --model gemma4:e4b-it-bf16-ctx128k
  ./agent_bench.py --target vllm-cv=$VLLM_CV_URL,model=google/gemma-4-E4B-it --runs 5 --json out.json

OpenCode is found via --opencode or OPENCODE_BIN, else in ./.tools (npm install --prefix .tools
opencode-ai). All agent state lives under ./.work, never in your home directory.
"""
import argparse
import glob
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, ".work")
TASKS = os.path.join(HERE, "tasks")
TEST_CMD = ["python3", "-m", "unittest", "discover", "-s", "tests", "-t", "."]


def normalize(url):
    if "://" not in url:
        url = "http://" + url
    return url.rstrip("/")


def parse_target(spec, default_model):
    name, _, rest = spec.partition("=")
    if not rest:
        sys.exit(f"--target needs NAME=URL, got {spec!r}")
    url, *opts = rest.split(",")
    model = default_model
    for o in opts:
        k, _, v = o.partition("=")
        if k == "model":
            model = v
        else:
            sys.exit(f"unknown target option {k!r} in {spec!r}")
    if not model:
        sys.exit(f"target {name}: no model (use --model or ,model=M)")
    return {"name": name, "url": normalize(url), "model": model}


def check_reachable(t):
    try:
        with urllib.request.urlopen(t["url"] + "/v1/models", timeout=10) as r:
            r.read()
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"target {t['name']}: {t['url']} not reachable ({e})")


def load_tasks(names):
    tasks = []
    for d in sorted(os.listdir(TASKS)):
        p = os.path.join(TASKS, d, "task.json")
        if os.path.isfile(p) and (not names or d in names):
            t = json.load(open(p))
            t["dir"] = os.path.join(TASKS, d)
            tasks.append(t)
    missing = set(names or []) - {t["id"] for t in tasks}
    if missing:
        sys.exit(f"unknown task(s): {', '.join(sorted(missing))}")
    return tasks


def prepare(task, target, n):
    run_dir = os.path.join(WORK, "runs", f"{target['name']}-{task['id']}-{n}")
    shutil.rmtree(run_dir, ignore_errors=True)
    shutil.copytree(os.path.join(task["dir"], "repo"), run_dir,
                    ignore=shutil.ignore_patterns("__pycache__"))
    git = ["git", "-c", "user.name=bench", "-c", "user.email=bench@localhost"]
    subprocess.run(["git", "init", "-q"], cwd=run_dir, check=True)
    subprocess.run(git + ["add", "-A"], cwd=run_dir, check=True)
    subprocess.run(git + ["commit", "-qm", "initial"], cwd=run_dir, check=True)
    return run_dir


def opencode_config(target, context, max_output):
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {"bench": {
            "npm": "@ai-sdk/openai-compatible", "name": "bench",
            "options": {"baseURL": target["url"] + "/v1"},
            "models": {target["model"]: {"name": target["model"],
                                         "limit": {"context": context, "output": max_output}}}}},
        "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny"},
    }


def run_opencode(opencode_bin, run_dir, target, task, args):
    """Run OpenCode headless; return (timed_out, wall_seconds, events)."""
    with open(os.path.join(run_dir, "opencode.json"), "w") as f:
        json.dump(opencode_config(target, args.context, args.max_output), f)
    xdg = os.path.join(WORK, "xdg")
    env = dict(os.environ)
    for var, sub in (("XDG_DATA_HOME", "data"), ("XDG_CONFIG_HOME", "config"),
                     ("XDG_CACHE_HOME", "cache"), ("XDG_STATE_HOME", "state")):
        env[var] = os.path.join(xdg, sub)
        os.makedirs(env[var], exist_ok=True)
    cmd = [opencode_bin, "run", "--model", f"bench/{target['model']}", "--format", "json",
           "--auto", task["prompt"]]
    t0 = time.perf_counter()
    timed_out = False
    # OpenCode takes its project directory from $PWD, not from the process's cwd. A child started
    # with cwd= still inherits the parent's PWD, so OpenCode would run in the wrong directory
    # (and fail with "Model not found", as the run's opencode.json is never loaded).
    env["PWD"] = run_dir
    proc = subprocess.Popen(cmd, cwd=run_dir, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        stdout, _ = proc.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, _ = proc.communicate()
    wall = time.perf_counter() - t0
    with open(os.path.join(run_dir, ".agent-output.txt"), "w") as f:
        f.write(stdout or "")
    events = []
    for line in (stdout or "").splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    return timed_out, wall, events


def summarize_events(events):
    tools, steps, tok_in, tok_out, text = {}, 0, 0, 0, []
    for e in events:
        part = e.get("part", {})
        if e["type"] == "tool_use":
            tools[part.get("tool", "?")] = tools.get(part.get("tool", "?"), 0) + 1
        elif e["type"] == "step_finish":
            steps += 1
            tk = part.get("tokens", {})
            tok_in += tk.get("input", 0) + tk.get("cache", {}).get("read", 0) + tk.get("cache", {}).get("write", 0)
            tok_out += tk.get("output", 0) + tk.get("reasoning", 0)
        elif e["type"] == "text":
            text.append(part.get("text", ""))
    return {"steps": steps, "tool_calls": sum(tools.values()), "tools": tools,
            "tokens_in": tok_in, "tokens_out": tok_out}, "\n".join(text)


def check(task, run_dir, answer):
    """Return (passed, reason)."""
    spec = task["check"]
    if spec.get("tests"):
        if subprocess.run(["git", "diff", "--quiet", "--", "tests"], cwd=run_dir).returncode != 0:
            return False, "tests were modified"
        try:
            r = subprocess.run(TEST_CMD, cwd=run_dir, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            return False, "tests hung"
        if r.returncode != 0:
            return False, "tests fail"
    if "forbid" in spec:
        rx = re.compile(spec["forbid"]["regex"])
        for path in glob.glob(os.path.join(run_dir, spec["forbid"]["glob"])):
            if rx.search(open(path).read()):
                return False, f"{os.path.relpath(path, run_dir)} still matches {spec['forbid']['regex']}"
    if "answer_contains" in spec:
        low = answer.lower()
        missing = [k for k in spec["answer_contains"] if k.lower() not in low]
        if missing:
            return False, f"answer lacks {missing}"
        if subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=run_dir,
                          capture_output=True, text=True).stdout.strip():
            return False, "files were modified"
    return True, "ok"


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", action="append", required=True, help="NAME=URL[,model=M] (repeatable)")
    p.add_argument("--model", help="model name for targets without ,model=")
    p.add_argument("--agent", choices=["opencode"], default="opencode")
    p.add_argument("--opencode", default=os.environ.get("OPENCODE_BIN") or
                   os.path.join(HERE, ".tools", "node_modules", ".bin", "opencode"))
    p.add_argument("--tasks", nargs="+", help="task ids (default: all in ./tasks)")
    p.add_argument("--runs", type=int, default=3, help="runs per (target, task)")
    p.add_argument("--timeout", type=int, default=900, help="seconds per run")
    p.add_argument("--context", type=int, default=131072, help="context window declared to the agent")
    p.add_argument("--max-output", type=int, default=8192, help="output limit declared to the agent")
    p.add_argument("--json", help="write raw results to this file")
    args = p.parse_args()

    targets = [parse_target(s, args.model) for s in args.target]
    tasks = load_tasks(args.tasks)
    if not os.path.isfile(args.opencode):
        sys.exit(f"opencode not found at {args.opencode} (install: npm install --prefix .tools opencode-ai)")
    for t in targets:
        check_reachable(t)

    results = []
    for t in targets:
        for task in tasks:
            for n in range(args.runs):
                run_dir = prepare(task, t, n)
                timed_out, wall, events = run_opencode(args.opencode, run_dir, t, task, args)
                stats, answer = summarize_events(events)
                passed, reason = (False, "timed out") if timed_out else check(task, run_dir, answer)
                r = {"target": t["name"], "model": t["model"], "task": task["id"], "run": n,
                     "passed": passed, "reason": reason, "timed_out": timed_out,
                     "wall": wall, **stats}
                results.append(r)
                print(f"{t['name']:<12} {task['id']:<10} run {n}: {'PASS' if passed else 'FAIL':<4} "
                      f"{wall:6.1f}s steps {stats['steps']:>2} tools {stats['tool_calls']:>2} "
                      f"tok in/out {stats['tokens_in']}/{stats['tokens_out']}  ({reason})", flush=True)

    print("\n=== summary (median over runs) ===")
    print(f"{'target':<12} {'task':<10} {'pass':>6} {'wall s':>8} {'steps':>6} {'tools':>6} {'tok in':>9} {'tok out':>8}")
    for t in targets:
        for task in tasks:
            rs = [r for r in results if r["target"] == t["name"] and r["task"] == task["id"]]
            ok = sum(r["passed"] for r in rs)
            print(f"{t['name']:<12} {task['id']:<10} {ok:>3}/{len(rs):<2} {median([r['wall'] for r in rs]):8.1f} "
                  f"{median([r['steps'] for r in rs]):6.0f} {median([r['tool_calls'] for r in rs]):6.0f} "
                  f"{median([r['tokens_in'] for r in rs]):9.0f} {median([r['tokens_out'] for r in rs]):8.0f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"agent": args.agent, "targets": targets, "runs": results}, f, indent=2)


if __name__ == "__main__":
    main()
