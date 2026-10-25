# Serving Gemma 4 with vllm-metal (Apple Silicon)

Notes from getting `vllm serve` working for Gemma 4 on a 64 GB Apple Silicon Mac
(vllm-metal 0.31.0, Python 3.12, macOS, Homebrew).

## Install

```bash
brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
brew install vllm-project/vllm-metal/vllm-metal
```

## Hugging Face token

Gemma models are gated: accept the license on the model page, then export a token
before serving. Keep it in a file outside the repo (here `~/.envvars`) and load it:

```bash
set -a && source ~/.envvars && set +a   # must define HF_TOKEN
```

Never commit the token.

## Problem 1: `google/gemma-4-26B-A4B` does not fit in memory

```bash
vllm serve google/gemma-4-26B-A4B
```

fails with:

```
ValueError: Paged attention: not enough Metal memory for KV cache.
metal_limit=55.66GB, fraction=0.92, usable_metal=51.21GB,
model_memory=50.47GB, overhead=1.22GB, kv_budget=-0.47GB.
```

The bf16 weights (~50 GB) fill the ~51 GB of Metal memory vLLM may use on a
64 GB machine, leaving nothing for the KV cache. Options:

- Use a 4-bit MLX build (~15-18 GB): `mlx-community/gemma-4-26B-A4B-it-qat-4bit`.
  **This works**, see "Serving the 26B model" below.
- Raise `--gpu-memory-utilization` (e.g. 0.98) and lower `--max-model-len`.
  Untested; leaves only ~2 GB for KV cache.
- Use a smaller model (what we did, below).

## What worked: a smaller model

```bash
set -a && source ~/.envvars && set +a
vllm serve google/gemma-4-E4B-it --host 0.0.0.0 --port 8000
```

- `--host 0.0.0.0` makes it listen on all interfaces (default is localhost only).
- Context length defaults to 131072 for this model; no `--max-model-len` needed.
- Check it:

```bash
curl localhost:8000/v1/models
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"google/gemma-4-E4B-it","messages":[{"role":"user","content":"Say hi"}],"max_tokens":20}'
```

There is no authentication. Add `--api-key <key>` if the port is reachable by
anyone else.

## Serving the 26B model (works): 4-bit MLX, tool calling, 131072 context

```bash
set -a && source ~/.envvars && set +a
vllm serve mlx-community/gemma-4-26B-A4B-it-qat-4bit \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 131072 \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4
```

- Repo `mlx-community/gemma-4-26B-A4B-it-qat-4bit` (MLX, QAT, 4-bit). The first start downloads it.
- vllm-metal 0.31.0 accepts the tool-calling flags and the `gemma4` parsers.
  `./tool_check.py --url http://127.0.0.1:8000` passes all 4 checks (single call, round trip,
  streaming, tool choice), so this host can be an agent backend.
- Serves text only: vllm-metal logs "Gemma 4 vision disabled" for a repo id that is not a local
  checkpoint directory.
- **No sleep mode.** `--enable-sleep-mode` fails at startup with
  `Sleep mode is not supported on current platform`. A true cold start on this host means
  stopping and restarting `vllm serve`.
- `--host 127.0.0.1` because other machines are blocked by the firewall (Problem 3); reach it
  through an SSH tunnel.

## Problem 2: `OSError: [Errno 48] Address already in use`

A previous `vllm serve` was still holding port 8000. `pkill -f "vllm serve"` did
not stop it because the process shows up as `.../Python.app/... /opt/homebrew/bin/vllm serve ...`.
Find and kill it explicitly:

```bash
pgrep -fl vllm
lsof -nP -iTCP:8000 -sTCP:LISTEN
kill <pid>
```

## Problem 3: server works locally but other machines get no response

`localhost:8000` returns 200, but `http://<mac-ip>:8000` gives
"Empty reply from server". The macOS application firewall had an explicit
"Block incoming connections" rule for the Homebrew Python.app that runs vLLM:

```bash
/usr/libexec/ApplicationFirewall/socketfilterfw --getappblocked \
  /opt/homebrew/Cellar/python@3.12/<version>/Frameworks/Python.framework/Versions/3.12/Resources/Python.app
```

The fix would be `socketfilterfw --unblockapp <that path>` (needs sudo), but on a
managed Mac firewall settings cannot be changed from the command line. The rule is
also tied to the Cellar path, so a Homebrew Python upgrade can bring it back.

Workaround: an SSH tunnel opened **from the client**. sshd on the Mac is reachable
(port 22 is not blocked), and the forwarded connection reaches vLLM on the Mac's own
loopback, so the Python firewall rule does not apply:

```bash
# run on the client machine; <mac-ip> is this Mac's LAN address
ssh -N -4 -L 127.0.0.1:8000:localhost:8000 <user>@<mac-ip>
# on the client: http://127.0.0.1:8000/v1
```

`-4` and the explicit `127.0.0.1` avoid `bind [::1]:8000: Cannot assign requested address`
on clients without an IPv6 loopback. Use `127.0.0.1`, not `localhost`, in the client URL for
the same reason. A reverse tunnel (`ssh -R`, started from the Mac) also avoids the firewall
but was not tried.
