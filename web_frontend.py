from __future__ import annotations

import argparse
import logging
import os
import threading
from dataclasses import dataclass

from flask import Flask, Response, jsonify, render_template_string

from audio_streaming import AudioStreamingEngine
from orchestrator import FullDuplexOrchestrator
from ui_events import UIEventBroker, encode_sse


PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Speech Pro Console</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #151923;
      --muted: #697386;
      --line: #d9dee8;
      --accent: #0f766e;
      --user: #e9f5ff;
      --assistant: #eef8f1;
      --warn: #fff4d6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font: 15px/1.45 "Segoe UI", system-ui, -apple-system, sans-serif;
    }
    header {
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 28px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 { margin: 0; font-size: 18px; font-weight: 650; }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      gap: 18px;
      width: min(1120px, calc(100vw - 32px));
      margin: 18px auto;
    }
    section, aside {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 0;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 11px;
      border: 1px solid var(--line);
      border-radius: 999px;
      color: var(--muted);
      font-size: 13px;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent);
    }
    .conversation {
      height: calc(100vh - 112px);
      display: flex;
      flex-direction: column;
    }
    .messages {
      flex: 1;
      overflow: auto;
      padding: 18px;
    }
    .empty {
      color: var(--muted);
      display: grid;
      height: 100%;
      place-items: center;
      text-align: center;
    }
    .msg {
      max-width: 78%;
      margin: 0 0 12px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .msg.user {
      margin-left: auto;
      background: var(--user);
    }
    .msg.assistant {
      background: var(--assistant);
    }
    .role {
      display: block;
      margin-bottom: 4px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
      text-transform: uppercase;
    }
    aside {
      height: calc(100vh - 112px);
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .side-head {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      font-weight: 650;
    }
    .steps {
      overflow: auto;
      padding: 10px 14px 16px;
    }
    .step {
      display: grid;
      grid-template-columns: 74px 1fr;
      gap: 10px;
      padding: 9px 0;
      border-bottom: 1px solid #eef1f6;
      color: var(--muted);
      font-size: 13px;
    }
    .step strong {
      color: var(--ink);
      font-weight: 600;
    }
    .step.barge_in {
      background: var(--warn);
      margin: 0 -8px;
      padding: 9px 8px;
      border-radius: 6px;
      border-bottom: 0;
    }
    @media (max-width: 800px) {
      header { padding: 0 16px; }
      main { grid-template-columns: 1fr; width: calc(100vw - 20px); }
      .conversation, aside { height: auto; min-height: 320px; }
      aside { max-height: 320px; }
      .msg { max-width: 92%; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Speech Pro Console</h1>
    <div class="status"><span class="dot"></span><span id="status">Starting</span></div>
  </header>
  <main>
    <section class="conversation">
      <div id="messages" class="messages">
        <div id="empty" class="empty">Speak into the microphone. Parsed messages will appear here.</div>
      </div>
    </section>
    <aside>
      <div class="side-head">Steps</div>
      <div id="steps" class="steps"></div>
    </aside>
  </main>
  <script>
    const messages = document.getElementById("messages");
    const steps = document.getElementById("steps");
    const statusEl = document.getElementById("status");
    const empty = document.getElementById("empty");
    const maxSteps = 18;

    function label(kind) {
      return {
        status: "State",
        message: "Parsed",
        barge_in: "Barge-in",
        warning: "Notice"
      }[kind] || kind;
    }

    function addStep(event) {
      if (event.kind === "message") return;
      const item = document.createElement("div");
      item.className = `step ${event.kind}`;
      item.innerHTML = `<strong>${label(event.kind)}</strong><span>${event.message}</span>`;
      steps.prepend(item);
      while (steps.children.length > maxSteps) steps.lastChild.remove();
    }

    function addMessage(event) {
      if (event.kind !== "message") return;
      empty.style.display = "none";
      const item = document.createElement("div");
      item.className = `msg ${event.role || "assistant"}`;
      const role = event.role === "user" ? "You" : "Assistant";
      item.innerHTML = `<span class="role">${role}</span>${escapeHtml(event.message)}`;
      messages.appendChild(item);
      messages.scrollTop = messages.scrollHeight;
    }

    function escapeHtml(text) {
      return text.replace(/[&<>"']/g, char => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      }[char]));
    }

    function handle(event) {
      if (event.kind === "status") statusEl.textContent = event.message;
      addMessage(event);
      addStep(event);
    }

    fetch("/api/events")
      .then(response => response.json())
      .then(events => events.forEach(handle));

    const source = new EventSource("/events");
    source.onmessage = message => handle(JSON.parse(message.data));
    ["status", "message", "barge_in", "warning"].forEach(name => {
      source.addEventListener(name, message => handle(JSON.parse(message.data)));
    });
  </script>
</body>
</html>
"""


@dataclass(frozen=True)
class WebConfig:
    llm_model: str
    whisper_model: str
    whisper_device: str | None
    tts_backend: str
    tts_voice: str
    input_device: int | None
    output_device: int | None
    output_rate: int | None


def create_app(config: WebConfig) -> Flask:
    app = Flask(__name__)
    broker = UIEventBroker()
    state = {"orchestrator": None, "started": False}
    state_lock = threading.RLock()

    def publish(kind: str, message: str, role: str | None = None) -> None:
        broker.publish(kind, message, role)

    def ensure_started() -> None:
        with state_lock:
            if state["started"]:
                return
            publish("status", "Starting")
            orchestrator = FullDuplexOrchestrator(
                llm_model=config.llm_model,
                whisper_model=config.whisper_model,
                whisper_device=config.whisper_device,
                tts_voice=config.tts_voice,
                tts_backend=config.tts_backend,
                input_device_index=config.input_device,
                output_device_index=config.output_device,
                output_rate=config.output_rate,
                vad_debug=False,
                event_sink=publish,
            )
            state["orchestrator"] = orchestrator
            state["started"] = True
            orchestrator.start_background()

    @app.get("/")
    def index():
        ensure_started()
        return render_template_string(PAGE)

    @app.get("/api/events")
    def event_snapshot():
        return jsonify(broker.snapshot())

    @app.get("/events")
    def events():
        ensure_started()
        subscriber = broker.subscribe()

        def stream():
            try:
                while True:
                    event = subscriber.get()
                    yield encode_sse(event)
            finally:
                broker.unsubscribe(subscriber)

        return Response(stream(), mimetype="text/event-stream")

    @app.post("/api/stop")
    def stop():
        with state_lock:
            orchestrator = state.get("orchestrator")
            if orchestrator:
                orchestrator.stop()
            state["started"] = False
            publish("status", "Stopped")
        return jsonify({"ok": True})

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flask dashboard for Speech Pro.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--llm-model", default="llama3.2:3b")
    parser.add_argument("--whisper-model", default="tiny")
    parser.add_argument("--whisper-device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--tts-backend", default="pyttsx3", choices=["auto", "edge", "pyttsx3"])
    parser.add_argument("--tts-voice", default="en-US-ChristopherNeural")
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument("--output-device", type=int, default=None)
    parser.add_argument("--output-rate", type=int, default=None)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
    )
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    if args.list_devices:
        for device in AudioStreamingEngine.list_devices():
            print(
                f"{device['index']:>2}: in={device['inputs']} out={device['outputs']} "
                f"rate={device['default_rate']}  {device['name']}"
            )
        return 0
    app = create_app(
        WebConfig(
            llm_model=args.llm_model,
            whisper_model=args.whisper_model,
            whisper_device=args.whisper_device,
            tts_backend=args.tts_backend,
            tts_voice=args.tts_voice,
            input_device=args.input_device,
            output_device=args.output_device,
            output_rate=args.output_rate,
        )
    )
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
