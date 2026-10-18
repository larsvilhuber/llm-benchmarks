#!/usr/bin/env python3
"""Check OpenAI-style tool calling on an endpoint (what coding agents such as OpenCode rely on).

  ./tool_check.py --url http://127.0.0.1:8000 [--model M] [--json out.json]

Three checks, temperature 0:
  1. single tool call: the model must call read_file with path "src/app.py"
  2. round trip: the tool result is sent back; the answer must use it
  3. streaming: same as 1 with stream=true; tool-call deltas must reassemble to valid JSON
  4. choice: with two tools, a question that needs the other one (run_shell)
Python 3 standard library only.
"""
import argparse
import json
import sys
import urllib.request

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a file from the repository and return its contents.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "path relative to the repo root"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": "Run a shell command in the repository and return stdout.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}},
]
SYSTEM = "You are a coding agent. Use the tools to inspect the repository; do not guess file contents."
ASK_READ = "What does the function main() in src/app.py return? Read the file first."
ASK_SHELL = "How many Python files are in the repository? Use a shell command to find out."
FILE = "def main():\n    return 42\n"


def post(url, body, stream=False):
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=600)
    if not stream:
        return json.load(r)
    calls, content = {}, ""
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
            continue
        for ch in json.loads(line[5:]).get("choices", []):
            d = ch.get("delta", {})
            content += d.get("content") or ""
            for tc in d.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                c["id"] = tc.get("id") or c["id"]
                f = tc.get("function", {})
                c["name"] += f.get("name") or ""
                c["arguments"] += f.get("arguments") or ""
    return {"content": content, "calls": [calls[k] for k in sorted(calls)]}


def body(model, messages, **kw):
    return {"model": model, "messages": messages, "tools": TOOLS, "temperature": 0,
            "max_tokens": 512, **kw}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True)
    p.add_argument("--model")
    p.add_argument("--json")
    a = p.parse_args()
    url = a.url.rstrip("/")
    model = a.model or json.load(urllib.request.urlopen(url + "/v1/models"))["data"][0]["id"]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ASK_READ}]
    res = {}

    # 1. single call
    m = post(url, body(model, msgs))["choices"][0]["message"]
    tc = (m.get("tool_calls") or [None])[0]
    args = json.loads(tc["function"]["arguments"]) if tc else {}
    ok = bool(tc) and tc["function"]["name"] == "read_file" and args.get("path", "").endswith("src/app.py")
    res["single_call"] = {"ok": ok, "tool_calls": m.get("tool_calls"), "content": m.get("content")}

    # 2. round trip
    if tc:
        msgs2 = msgs + [{"role": "assistant", "content": m.get("content") or "", "tool_calls": [tc]},
                        {"role": "tool", "tool_call_id": tc["id"], "content": FILE}]
        m2 = post(url, body(model, msgs2))["choices"][0]["message"]
        res["round_trip"] = {"ok": "42" in (m2.get("content") or ""), "content": m2.get("content"),
                             "tool_calls": m2.get("tool_calls")}
    else:
        res["round_trip"] = {"ok": False, "content": "skipped: no tool call in step 1"}

    # 3. streaming
    s = post(url, body(model, msgs, stream=True), stream=True)
    try:
        sargs = json.loads(s["calls"][0]["arguments"]) if s["calls"] else {}
    except json.JSONDecodeError:
        sargs = None
    res["streaming"] = {"ok": bool(s["calls"]) and s["calls"][0]["name"] == "read_file"
                        and isinstance(sargs, dict) and sargs.get("path", "").endswith("src/app.py"),
                        "calls": s["calls"], "content": s["content"]}

    # 4. choosing the other tool
    m4 = post(url, body(model, [{"role": "system", "content": SYSTEM},
                                {"role": "user", "content": ASK_SHELL}]))["choices"][0]["message"]
    tc4 = (m4.get("tool_calls") or [None])[0]
    res["tool_choice"] = {"ok": bool(tc4) and tc4["function"]["name"] == "run_shell",
                          "tool_calls": m4.get("tool_calls"), "content": m4.get("content")}

    for k, v in res.items():
        print(f"{k:<12} {'PASS' if v['ok'] else 'FAIL'}  "
              f"{json.dumps(v.get('tool_calls') or v.get('calls') or v.get('content'))[:150]}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"url": url, "model": model, "results": res}, f, indent=2)
    sys.exit(0 if all(v["ok"] for v in res.values()) else 1)


if __name__ == "__main__":
    main()
