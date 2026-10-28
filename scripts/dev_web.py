"""Serve web/ locally with the same cross-origin-isolation headers Vercel will send (needed for multi-threaded WASM).

    python scripts/dev_web.py [port]      ->  http://127.0.0.1:8090
Also exposes data/ood as /_samples/ for parity tests (never deployed).
"""
import functools
import http.server
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm",
                      ".onnx": "application/octet-stream"}

    def translate_path(self, path):
        if path.startswith("/_samples/"):
            return str(ROOT / "data" / "ood" / path[len("/_samples/"):].split("?")[0])
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
    handler = functools.partial(Handler, directory=str(ROOT / "web"))
    print(f"http://127.0.0.1:{port}", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
