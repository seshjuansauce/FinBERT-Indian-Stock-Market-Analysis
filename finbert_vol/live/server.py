"""HTTP server inside the Cloudflare container (port 8080).

  GET  /       → "ok" (readiness probe the Durable Object polls)
  POST /run    body: state.tar.gz (may be empty on the very first run)
               → JSON {"ok", "log", "latest", "public": {name: text}, "state_b64": base64 tar.gz}
The Worker writes `public` files and the new state back to R2. Nothing persists in the container.
"""
from __future__ import annotations
import base64
import json
import os
import tempfile
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models/finbert-in-sentiment")
CALIBRATION = os.environ.get("CALIBRATION", "/app/results/sentiment/calibration.json")
_model = None


def sentiment_model():
    global _model
    if _model is None:
        from finbert_vol.models import SentimentModel
        _model = SentimentModel.from_settings(MODEL_DIR, CALIBRATION, "C", "cpu")
    return _model


def run(blob: bytes, model=None, prices=None, news=None) -> dict:
    from .runner import Runner, State
    lines = []
    with tempfile.TemporaryDirectory() as tmp:
        st = State.unpack(blob, Path(tmp) / "state")
        r = Runner(st, model or sentiment_model(), prices=prices, news=news, log=lambda m: (print(m, flush=True), lines.append(str(m))))
        try:
            latest = r.daily(); ok, err = True, None
        except Exception as e:                            # keep yesterday's outputs; report the error
            ok, err, latest = False, f"{type(e).__name__}: {e}", None
            lines.append(traceback.format_exc()); print(traceback.format_exc(), flush=True)
        pub = {p.name: p.read_text() for p in (st.root / "public").glob("*") if p.is_file()}
        return {"ok": ok, "error": err, "log": lines, "latest": latest, "public": pub,
                "state_b64": base64.b64encode(st.pack()).decode() if ok else None}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype="application/json"):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)

    def do_GET(self): self._send(200, b"ok", "text/plain")

    def do_POST(self):
        if self.path != "/run": return self._send(404, b'{"error":"not found"}')
        n = int(self.headers.get("Content-Length") or 0); blob = self.rfile.read(n) if n else b""
        res = run(blob)
        self._send(200 if res["ok"] else 500, json.dumps(res).encode())

    def log_message(self, fmt, *args): print("[http] " + fmt % args, flush=True)


def serve(port: int = 8080):
    print(f"listening on :{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
