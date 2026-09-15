"""Flask server that reads snapshots produced by the perception loop."""
from __future__ import annotations

import threading
from pathlib import Path

from flask import Flask, Response, jsonify, render_template
from werkzeug.serving import BaseWSGIServer, make_server

from shared_state import SharedState


class DashboardServer:
    def __init__(self, state: SharedState, host: str = "0.0.0.0", port: int = 3000):
        root = Path(__file__).resolve().parent
        self.state = state
        self.app = Flask(
            __name__, template_folder=str(root / "templates"), static_folder=str(root / "static")
        )
        self._server: BaseWSGIServer | None = None
        self._thread: threading.Thread | None = None
        self.host = host
        self.port = port
        self._register_routes()

    def _register_routes(self) -> None:
        @self.app.get("/")
        def index():
            return render_template("index.html")

        @self.app.get("/api/status")
        def status():
            return jsonify(self.state.snapshot())

        @self.app.get("/healthz")
        def health():
            snapshot = self.state.snapshot()
            code = 200 if snapshot["camera"] == "running" else 503
            return jsonify({"status": snapshot["camera"]}), code

        @self.app.get("/video_feed")
        def video_feed():
            return Response(
                self._mjpeg_stream(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
                headers={"Cache-Control": "no-store"},
            )

    def _mjpeg_stream(self):
        version = -1
        while True:
            version, jpeg = self.state.wait_for_jpeg(version)
            if jpeg is None:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    def start(self) -> None:
        self._server = make_server(self.host, self.port, self.app, threaded=True)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="dashboard", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
