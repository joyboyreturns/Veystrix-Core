"""Remember each client's static prompt prefix (system message + tool schemas) and pre-cache it for every reasoning effort.

With the default ``VEYSTRIX_EFFORT_PLACEMENT=late`` (veystrix/effort_slot.py) every effort shares the cached tools + system
prompt, so one effort's prefill already pays for all the others: a second effort re-prefills only its short effort turn and
the conversation after it. Warming still earns its keep: it lays down the first copy when nothing is cached yet (a new client, a changed system prompt, an evicted entry), and refreshes the
entries' last-use time so the SSD budget does not evict a prompt you still use. With ``=template`` (the checkpoint's own
layout, effort sentence first) the efforts share nothing and warming is the only way to avoid a cold prefill per effort.

``./serve.sh --warm`` sends one request per client and effort, one generated token each, through the normal API (single slot,
swap guard active), stopping at the first safety stop and never retrying. ``./serve.sh --overnight`` does the same by itself
while the server has been idle and the Mac is on AC power (``models/flashnext/overnight.py``, ``AutoWarmer``).

What is stored: ``~/.flashnext/warm/<client>.json`` (0600) holds the system message and the tool schemas the client sent. That
text can name your directories; it never leaves this machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

WARM_DIR = Path(os.path.expanduser(os.environ.get("VEYSTRIX_WARM_DIR", "~/.flashnext/warm")))
WARM_HEADER = "X-Veystrix-Warm"
DEFAULT_EFFORTS = ("low", "medium", "high", "xhigh")     # off == medium (same prefix), max -> xhigh
WARM_USER_TEXT = "ready?"


def family(user_agent: str) -> str:
    """'QwenCode/0.27.0 (darwin; arm64)' -> 'qwencode'; empty or odd agents -> 'unknown'."""
    head = re.split(r"[/\s(]", (user_agent or "").strip(), maxsplit=1)[0].lower()
    head = re.sub(r"[^a-z0-9._-]", "", head)
    return head[:40] or "unknown"


def _digest(system, tools) -> str:
    return hashlib.sha256(json.dumps([system, tools], sort_keys=True, default=str).encode()).hexdigest()


def remember(user_agent: str, messages, tools, directory=None) -> bool:
    """Store the latest static prefix of a tool-using client. Returns True when the file changed. Never raises."""
    try:
        if not tools or not messages or messages[0].get("role") != "system":
            return False
        directory = Path(directory or WARM_DIR)
        name = family(user_agent)
        system = messages[0]
        digest = _digest(system, tools)
        path = directory / f"{name}.json"
        try:
            if json.loads(path.read_text()).get("digest") == digest:
                return False
        except (OSError, ValueError):
            pass
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump({"family": name, "user_agent": user_agent[:120], "saved": time.time(), "digest": digest,
                       "system": system, "tools": tools}, handle)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def load(directory=None) -> list:
    out = []
    for path in sorted(Path(directory or WARM_DIR).glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if data.get("system") and data.get("tools"):
            out.append(data)
    return out


def request_body(entry: dict, effort: str, model_id: str = "veystrix-flashnext") -> dict:
    return {"model": model_id, "stream": True, "reasoning_effort": effort,
            "messages": [entry["system"], {"role": "user", "content": WARM_USER_TEXT}], "tools": entry["tools"]}


def http_client_errors():
    import http.client
    return http.client.HTTPException


class WarmStop(Exception):
    """The server said stop (memory safety stop, not ready): the run ends, nothing is retried."""


class WarmAborted(Exception):
    """A real request arrived while a warm-up was running; the warm-up was dropped so the request goes first."""


STATE_NAME = "state.json"
REFRESH_SECONDS = 6 * 3600.0        # a variant warmed (or touched) within this long is left alone


def job_key(entry: dict, effort: str) -> str:
    return f"{entry['family']}:{str(entry.get('digest', ''))[:12]}:{effort}"


def load_state(directory=None) -> dict:
    try:
        return json.loads((Path(directory or WARM_DIR) / STATE_NAME).read_text())
    except (OSError, ValueError):
        return {}


def save_state(state: dict, directory=None) -> None:
    try:
        directory = Path(directory or WARM_DIR)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        tmp = directory / (STATE_NAME + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(state, handle)
        os.replace(tmp, directory / STATE_NAME)
    except OSError:
        pass


def due_jobs(entries, efforts, state, now, refresh_seconds=REFRESH_SECONDS) -> list:
    """(entry, effort) pairs never warmed, or not within ``refresh_seconds``. A changed system prompt is a new key."""
    return [(entry, effort) for entry in entries for effort in efforts
            if now - float(state.get(job_key(entry, effort), 0.0)) > refresh_seconds]


def _get_json(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def warm_one(base: str, entry: dict, effort: str, *, key: str, timeout: float, abort=None,
             model_id: str = "veystrix-flashnext") -> dict:
    """One prefill-only request. Returns {'seconds', 'hit_tokens', 'prefilled_tokens', 'outcome', 'prompt_tokens'}."""
    body = json.dumps(request_body(entry, effort, model_id)).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", body, {
        "Content-Type": "application/json", "Authorization": f"Bearer {key}", WARM_HEADER: "1",
        "User-Agent": entry.get("user_agent") or "veystrix-warm"})
    began = time.time()
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:300].decode(errors="replace")
        if exc.code == 503 and "memory_critical_stop" in detail:
            raise WarmStop(f"memory safety stop: {detail}") from None
        raise WarmStop(f"HTTP {exc.code}: {detail}") from None
    error = None
    finished = threading.Event()
    if abort is not None:
        def watch():
            while not finished.is_set():
                if abort.wait(0.5):
                    try:
                        response.fp.raw._sock.shutdown(socket.SHUT_RDWR)     # the server cancels a prefill whose client is gone
                    except Exception:
                        pass
                    return
        threading.Thread(target=watch, daemon=True).start()
    try:
        with response:
            for raw in response:
                line = raw.strip()
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                try:
                    event = json.loads(line[6:])
                except ValueError:
                    continue
                if "error" in event:
                    error = event["error"]
    except (OSError, ValueError, http_client_errors()):
        if abort is not None and abort.is_set():
            raise WarmAborted() from None
        raise
    finally:
        finished.set()
    if abort is not None and abort.is_set():
        raise WarmAborted()
    if error is not None:
        message = str(error.get("message", error)) if isinstance(error, dict) else str(error)
        if "memory_critical_stop" in message or "safety stop" in message.lower():
            raise WarmStop(f"memory safety stop: {message}")
        # The one token the warm request asked for was the start of a tool call: the prefill, which is the point, is done.
        if "output limit of 1 tokens" not in message:
            raise WarmStop(f"server error: {message[:300]}")
    last = {}
    try:
        last = (_get_json(base + "/v1/cache/diagnostics").get("last_prefix") or {})
    except Exception:
        pass
    return {"seconds": round(time.time() - began, 1), "prompt_tokens": last.get("prompt_tokens"),
            "hit_tokens": last.get("hit_tokens"), "prefilled_tokens": last.get("prefilled_tokens"), "outcome": last.get("outcome")}


def build_parser():
    p = argparse.ArgumentParser(description="Pre-cache each client's system prompt for every reasoning effort (idle-time job).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.environ.get("VEYSTRIX_PORT", "8080")))
    p.add_argument("--efforts", default=",".join(DEFAULT_EFFORTS), help="comma list from low,medium,high,xhigh")
    p.add_argument("--client", action="append", help="only this client family (default: every remembered client)")
    p.add_argument("--last", help="effort warmed last, so it is the one the live conversation state holds afterwards")
    p.add_argument("--dry-run", action="store_true", help="list what would be sent and exit")
    p.add_argument("--timeout", type=float, default=3600.0)
    return p


def order_efforts(efforts, last=None):
    efforts = list(dict.fromkeys(efforts))
    if last in efforts:
        efforts.remove(last)
        efforts.append(last)
    return efforts


def main(argv=None, directory=None, out=print) -> int:
    args = build_parser().parse_args(argv)
    efforts = [e.strip() for e in args.efforts.split(",") if e.strip()]
    bad = [e for e in efforts if e not in DEFAULT_EFFORTS]
    if bad:
        out(f"unsupported effort {bad[0]!r}; use {', '.join(DEFAULT_EFFORTS)} (off is the same prefix as medium)")
        return 2
    efforts = order_efforts(efforts, args.last)
    entries = [e for e in load(directory) if not args.client or e["family"] in args.client]
    if not entries:
        out("nothing to warm yet: use Qwen Code or DeepSeek Harness against the server once; the server remembers their system "
            f"prompt and tool schemas in {directory or WARM_DIR}")
        return 1
    plan = [(e, effort) for e in entries for effort in efforts]
    out(f"warming {len(entries)} client(s) x {len(efforts)} effort(s) = {len(plan)} prefill(s): "
        + ", ".join(f"{e['family']}:{effort}" for e, effort in plan))
    if args.dry_run:
        return 0
    base = f"http://{args.host}:{args.port}"
    try:
        health = _get_json(base + "/health")
    except Exception as exc:
        out(f"the server is not reachable at {base}: {exc}")
        return 3
    if not health.get("ready") or health.get("status") != "ok":
        out(f"the server is not ready (status={health.get('status')}); not warming")
        return 3
    if (health.get("worker") or {}).get("busy"):
        out("the server is busy with a request: warm-up would queue behind it and replace its conversation state; try when idle")
        return 4
    key = os.environ.get("VEYSTRIX_API_KEY", "local")
    for entry, effort in plan:
        began = time.time()
        try:
            result = warm_one(base, entry, effort, key=key, timeout=args.timeout,
                              model_id=health.get("model") or "veystrix-flashnext")
        except WarmStop as exc:
            out(f"STOPPED at {entry['family']}:{effort} after {time.time() - began:.0f}s: {exc}. Nothing was retried.")
            return 5
        except Exception as exc:
            out(f"STOPPED at {entry['family']}:{effort}: {type(exc).__name__}: {exc}. Nothing was retried.")
            return 5
        state = load_state(directory)
        state[job_key(entry, effort)] = time.time()
        save_state(state, directory)
        out(f"  {entry['family']:<18} {effort:<7} {result['seconds']:>6}s  prompt {result['prompt_tokens']}  "
            f"from cache {result['hit_tokens']}  prefilled {result['prefilled_tokens']}  [{result['outcome']}]")
    out("done. Conversation state now holds the last variant; the others are on SSD and restore in about a second.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
