#!/usr/bin/env python3
"""
webgui/serve.py — a tiny stdlib web GUI for the local LLM stack.

A thin shell over machinery that already exists:
  • model list   → scripts/model_registry.iter_models()  (instruct families)
                 + a scan of playground/*/config.json     (base models)
  • serve/stop   → scripts/serve_local.sh + an lsof port-kill (same as the menu)
  • status       → HTTP health probes on the inference ports
  • chat / text  → a streaming proxy to :8080 (llama.cpp) / :8085 (MLX)

Stdlib only, so it runs before any project virtualenv is active. Launch with
`python3 llm.py web` or `python3 webgui/serve.py` and open the printed URL.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
PLAYGROUND = ROOT / "playground"
INDEX_HTML = Path(__file__).resolve().parent / "index.html"
SERVE_SH = SCRIPTS / "serve_local.sh"

sys.path.insert(0, str(SCRIPTS))
import model_registry  # noqa: E402  (after sys.path insert)

# Ports the inference backends listen on (mirrors llm.py).
LLAMACPP_PORT = 8080
MLX_PORT = 8085
OMLX_PORT = 8123
PROBE_PORTS = {
    LLAMACPP_PORT: "llama.cpp",
    MLX_PORT: "MLX (mlx_lm / mlx_vlm)",
    OMLX_PORT: "oMLX (always-on)",
}

# Backends the GUI can launch and the port each one binds.
BACKEND_PORT = {
    "llamacpp": LLAMACPP_PORT,
    "mlx": MLX_PORT,
    "mlx-kv": MLX_PORT,
    "mlx-vlm": MLX_PORT,
}


# ---------------------------------------------------------------------------
# Pure logic — model discovery (unit-tested in webgui/test_serve.py)
# ---------------------------------------------------------------------------

def port_for(backend: str) -> int:
    """Inference port a given launch backend binds to."""
    return BACKEND_PORT.get(backend, LLAMACPP_PORT)


def instruct_families() -> list[dict]:
    """Registry families collapsed to one row each, with on-disk backends.

    Mirrors llama_serve_menu._rows(): a backend is only offered if its weights
    are actually on disk, and an on-disk MLX family also gains the mlx-kv
    (mlx_vlm + KV q8 turboquant) option.
    """
    rows = model_registry.iter_models(model_registry.DEFAULT_CONFIG)
    families: dict[str, dict] = {}
    for row in rows:
        fid = row["family_id"]
        fam = families.get(fid)
        if fam is None:
            aliases = row.get("aliases", [])
            fam = families[fid] = {
                "id": fid,
                "type": "instruct",
                "label": aliases[0] if aliases else fid,
                "use_case": row.get("use_case", ""),
                "quant": row.get("quant", "?"),
                "backends": set(),
                "mtp_supported": False,
                "mlx_server": "mlx_lm",  # resolved MLX server, refined below
            }
        if not row.get("exists"):
            continue
        fam["backends"].add(row["backend"])
        if row["backend"] == "llamacpp" and row.get("mtp_supported"):
            fam["mtp_supported"] = True
        # Surface which MLX server actually serves this family so the UI can name
        # the exact backend. Most omni models run as text under mlx_lm; only the
        # registry-pinned ones (e4b, gemma4_unified 12B) need mlx_vlm.
        if row["backend"] == "mlx" and row.get("mlx_server") == "mlx_vlm":
            fam["mlx_server"] = "mlx_vlm"

    result = []
    for fam in families.values():
        # Only offer plain on-disk backends. mlx-kv (mlx_vlm + KV q8 turboquant)
        # is a niche variant — intentionally CLI-only (`llm serve --backend mlx-kv`)
        # to keep the picker uncluttered.
        fam["backends"] = sorted(fam["backends"])
        if fam["backends"]:  # hide families with nothing on disk
            result.append(fam)
    return sorted(result, key=lambda f: f["id"])


def base_models() -> list[dict]:
    """Base (non-instruct) MLX models discovered under playground/.

    These are not in the registry on purpose (standalone, not part of the
    benchmark suite). Any directory with a config.json is an MLX dir mlx_lm
    can load and serve via /v1/completions.
    """
    if not PLAYGROUND.is_dir():
        return []
    result = []
    for child in sorted(PLAYGROUND.iterdir()):
        config = child / "config.json"
        if not (child.is_dir() and config.is_file()):
            continue
        model_type = ""
        try:
            model_type = json.loads(config.read_text()).get("model_type", "")
        except (ValueError, OSError):
            pass
        result.append({
            "id": child.name,
            "type": "base",
            "label": child.name,
            "use_case": f"base model — {model_type}" if model_type else "base model",
            "path": str(child),
            "model_type": model_type,
            "backends": ["mlx"],
            "mtp_supported": False,
            "mlx_server": "mlx_lm",
        })
    return result


def model_catalog() -> dict:
    return {"instruct": instruct_families(), "base": base_models()}


def is_base_path(path: str) -> bool:
    """True only for an existing directory under playground/ (launch guard)."""
    try:
        resolved = Path(path).resolve()
        return resolved.is_dir() and PLAYGROUND.resolve() in resolved.parents
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Lifecycle — serve / stop / status
# ---------------------------------------------------------------------------

# Track servers this GUI launched: port -> Popen. serve_local.sh exec()s the
# real server, so the child IS the server; stop = kill by port regardless.
_CHILDREN: dict[int, subprocess.Popen] = {}

# Exact model path launched on each port, so the proxy can echo it back as the
# request "model". This matters because the MLX servers disagree on the field:
#   mlx_vlm.server  defaults to a built-in model (nanoLLaVA) when "model" is
#                   absent, then fails under HF_HUB_OFFLINE — it NEEDS the path.
#   mlx_lm.server   serves the loaded weights regardless, but /v1/models lists
#                   every cached model, so guessing the id picks the wrong one.
# The one value that satisfies both is the exact path we launched (verified:
# mlx_lm returns 200 for its own path, 404 for any other cached id).
_PORT_MODEL: dict[int, str] = {}

# --- idle watchdog state ---------------------------------------------------
# "Activity" is a launch, stop, or generation — NOT the browser's 4s status
# polls (those would keep the server alive forever with a tab open). When no
# activity for idle_timeout seconds AND nothing is mid-generation, the watchdog
# unloads the model and shuts the GUI down (full teardown).
_activity_lock = threading.Lock()
_last_activity = time.time()
_active_requests = 0


def _touch() -> None:
    global _last_activity
    with _activity_lock:
        _last_activity = time.time()


def _enter_request() -> None:
    global _active_requests, _last_activity
    with _activity_lock:
        _active_requests += 1
        _last_activity = time.time()


def _leave_request() -> None:
    global _active_requests, _last_activity
    with _activity_lock:
        _active_requests = max(0, _active_requests - 1)
        _last_activity = time.time()


def _kill_port(port: int) -> None:
    """Kill whatever is LISTENing on `port` (lifted from llama_serve_menu)."""
    try:
        out = subprocess.check_output(
            ["lsof", "-i", f"tcp:{port}", "-sTCP:LISTEN", "-t"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return
    for pid in (p for p in out.splitlines() if p.strip()):
        subprocess.run(["kill", pid], check=False)
    _CHILDREN.pop(port, None)
    _PORT_MODEL.pop(port, None)


def _probe(port: int) -> tuple[bool, str | None]:
    """(up, loaded_model_id). Tries /v1/models then /health."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as r:
            data = json.loads(r.read())
            models = data.get("data") or []
            return True, (models[0].get("id") if models else None)
    except Exception:
        pass
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
            return True, None
    except Exception:
        return False, None


def status() -> list[dict]:
    rows = []
    for port, label in PROBE_PORTS.items():
        up, loaded = _probe(port)
        rows.append({"port": port, "label": label, "up": up, "loaded": loaded})
    return rows


def serve_model(selector: str, backend: str, no_mtp: bool = False) -> dict:
    """Validate, free the target port, then launch serve_local.sh detached.

    no_mtp forces the plain GGUF baseline (passes --no-mtp), letting the UI
    turn off MTP speculative decoding even on models that carry an MTP head.
    """
    if backend not in BACKEND_PORT:
        raise ValueError(f"Unknown backend: {backend}")

    # Validate the selector against the registry (instruct) or playground (base).
    # Never launch an arbitrary path: base launches are restricted to playground/.
    # Capture the exact model path so the proxy can pin it as the request "model".
    if is_base_path(selector):
        if backend != "mlx":
            raise ValueError("Base models serve only via the mlx backend.")
        model_path = str(Path(selector).resolve())
    else:
        resolve_backend = "mlx" if backend in ("mlx", "mlx-kv", "mlx-vlm") else backend
        row = model_registry.resolve(selector, resolve_backend, model_registry.DEFAULT_CONFIG)
        model_path = row.get("path", "")

    port = port_for(backend)
    _kill_port(port)
    cmd = [str(SERVE_SH), selector, "--backend", backend]
    if no_mtp:
        cmd.append("--no-mtp")
    child = subprocess.Popen(cmd, cwd=str(ROOT))
    _CHILDREN[port] = child
    _PORT_MODEL[port] = model_path
    return {"ok": True, "port": port, "pid": child.pid, "backend": backend,
            "selector": selector, "no_mtp": no_mtp, "model_path": model_path}


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args) -> None:  # keep the console quiet
        pass

    # -- helpers ------------------------------------------------------------
    def _send_json(self, payload: object, code: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    # -- routing ------------------------------------------------------------
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._serve_index()
        if path == "/api/models":
            return self._send_json(model_catalog())
        if path == "/api/status":
            return self._send_json({"ports": status()})
        self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/serve":
            return self._handle_serve()
        if path == "/api/stop":
            return self._handle_stop()
        if path in ("/v1/chat/completions", "/v1/completions"):
            return self._proxy(path, parse_qs(parsed.query))
        self._send_json({"error": "not found"}, 404)

    # -- handlers -----------------------------------------------------------
    def _serve_index(self) -> None:
        try:
            body = INDEX_HTML.read_bytes()
        except OSError:
            return self._send_json({"error": "index.html missing"}, 500)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_serve(self) -> None:
        _touch()
        try:
            req = json.loads(self._read_body() or b"{}")
            result = serve_model(req["selector"], req["backend"], bool(req.get("no_mtp")))
            self._send_json(result)
        except (KeyError, ValueError, SystemExit) as exc:
            self._send_json({"ok": False, "error": str(exc)}, 400)

    def _handle_stop(self) -> None:
        _touch()
        try:
            req = json.loads(self._read_body() or b"{}")
            _kill_port(int(req["port"]))
            self._send_json({"ok": True})
        except (KeyError, ValueError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, 400)

    def _inject_model(self, body: bytes, port: int) -> bytes:
        """Pin the request to the exact model path this GUI launched on `port`.

        See _PORT_MODEL for why the launched path — not /v1/models or a default —
        is the only value that satisfies both mlx_lm and mlx_vlm. Only fills the
        field when the client left it blank; an explicit client "model" wins.
        """
        if not body:
            return body
        try:
            obj = json.loads(body)
        except ValueError:
            return body
        path = _PORT_MODEL.get(port)
        if path and not obj.get("model"):
            obj["model"] = path
            return json.dumps(obj).encode()
        return body

    def _proxy(self, path: str, query: dict) -> None:
        """Stream a chat/completions request through to the target backend."""
        _enter_request()
        try:
            self._proxy_inner(path, query)
        finally:
            _leave_request()  # also resets the idle clock on completion

    def _proxy_inner(self, path: str, query: dict) -> None:
        port = int(query.get("port", [LLAMACPP_PORT])[0])
        body = self._inject_model(self._read_body(), port)
        upstream = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(upstream, timeout=600)
        except urllib.error.HTTPError as exc:  # forward backend's own error body
            detail = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(detail)))
            self.end_headers()
            self.wfile.write(detail)
            return
        except urllib.error.URLError as exc:
            return self._send_json(
                {"error": f"backend on :{port} is not running ({exc.reason})"}, 502)

        # Stream the response straight through (SSE when the client asked to stream).
        self.send_response(200)
        ctype = resp.headers.get("Content-Type", "application/json")
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                chunk = resp.read(1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client navigated away mid-stream
        finally:
            resp.close()


class GuiServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that stays quiet on mid-stream client disconnects.

    A chat UI disconnects all the time — the user stops a generation or closes
    the tab — which surfaces as BrokenPipeError/ConnectionResetError when the
    handler's final buffer flush hits a closed socket. That's expected, not an
    error worth a traceback. Anything else still propagates.
    """
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def _unload_all_models() -> None:
    """Kill every inference server this GUI launched (frees the model RAM)."""
    for port in list(_PORT_MODEL) or [LLAMACPP_PORT, MLX_PORT]:
        _kill_port(port)


def _should_shutdown(timeout: int) -> bool:
    """True when nothing is mid-generation and we've been idle past `timeout`."""
    with _activity_lock:
        return _active_requests == 0 and (time.time() - _last_activity) >= timeout


def _idle_watchdog(server: ThreadingHTTPServer, timeout: int) -> None:
    """Unload the model and shut the GUI down after `timeout`s of no activity."""
    while True:
        time.sleep(min(15, timeout))
        if _should_shutdown(timeout):
            print(f"\nIdle for ~{timeout}s — unloading model and shutting down.")
            _unload_all_models()
            server.shutdown()  # unblocks serve_forever() in the main thread
            return


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _check_bind_host(host: str, allow_remote: bool) -> None:
    """This GUI can start and stop model servers and has no authentication.
    Binding it anywhere but loopback hands those controls to the whole network,
    so require an explicit opt-in flag rather than trusting --host alone."""
    if host in LOOPBACK_HOSTS or allow_remote:
        return
    raise SystemExit(
        f"Refusing to bind {host}: this GUI is unauthenticated and can launch, stop,\n"
        f"and proxy model servers. Anyone who can reach {host}:PORT would control them.\n"
        f"Use 127.0.0.1 (default), an SSH tunnel, or pass --allow-remote if you have\n"
        f"deliberately firewalled the port."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Local LLM web GUI (stdlib, zero deps)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-open", action="store_true", help="Do not open a browser")
    ap.add_argument("--allow-remote", action="store_true",
                    help="Permit binding a non-loopback host (unauthenticated: see docs)")
    ap.add_argument("--idle-timeout", type=int, default=300,
                    help="Seconds of inactivity before auto-unloading the model and "
                         "shutting down (0 disables). Default: 300 (5 min).")
    args = ap.parse_args()
    _check_bind_host(args.host, args.allow_remote)

    server = GuiServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    idle_note = f"auto-unload + shutdown after {args.idle_timeout}s idle" if args.idle_timeout > 0 else "idle timeout off"
    print(f"Local LLM GUI → {url}  (Ctrl+C to stop · {idle_note})")
    if args.idle_timeout > 0:
        threading.Thread(target=_idle_watchdog, args=(server, args.idle_timeout),
                         daemon=True).start()
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        _unload_all_models()
        server.shutdown()


if __name__ == "__main__":
    main()
