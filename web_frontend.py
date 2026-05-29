from __future__ import annotations

import argparse
import logging
import os
import threading
from dataclasses import dataclass

from flask import Flask, Response, jsonify, render_template_string, request

from audio_streaming import AudioStreamingEngine
from orchestrator import FullDuplexOrchestrator
from performance_metrics import PerformanceMetrics
from ui_events import UIEventBroker, encode_sse


PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Speech Pro Console</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=JetBrains+Mono:wght@500&display=swap');
    :root {
      color-scheme: light;
      --bg: #f3efe7;
      --bg-radial: radial-gradient(1200px 480px at 82% -10%, #ffd7a9 0%, rgba(255, 215, 169, 0.16) 50%, transparent 75%), radial-gradient(900px 420px at -10% 20%, #cfe3ff 0%, rgba(207, 227, 255, 0.22) 45%, transparent 72%);
      --panel: rgba(255, 255, 255, 0.86);
      --ink: #1c1f2a;
      --muted: #5d6578;
      --line: #d6d9e3;
      --accent: #0d7c66;
      --accent-soft: #dcf6ee;
      --blue: #1f5fe0;
      --blue-soft: #e7efff;
      --user: #eef6ff;
      --assistant: #eff9f2;
      --warn: #fff3d4;
      --shadow: 0 10px 30px rgba(22, 32, 61, 0.1);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg-radial), var(--bg);
      color: var(--ink);
      font: 15px/1.5 "Sora", "Segoe UI", system-ui, sans-serif;
    }
    header {
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 28px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(90deg, rgba(255,255,255,0.94), rgba(255,255,255,0.78));
      backdrop-filter: blur(10px);
    }
    h1 {
      margin: 0;
      font-size: 19px;
      font-weight: 700;
      letter-spacing: 0.2px;
    }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 430px;
      gap: 18px;
      width: min(1280px, calc(100vw - 32px));
      margin: 18px auto;
    }
    section, aside {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      min-height: 0;
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
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
      background: rgba(255, 255, 255, 0.7);
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
      border-radius: 12px;
      white-space: pre-wrap;
      word-break: break-word;
      box-shadow: 0 6px 16px rgba(27, 34, 63, 0.08);
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
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .side-head small {
      color: var(--muted);
      font-weight: 500;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      padding: 14px;
      border-bottom: 1px solid var(--line);
    }
    .metric {
      min-height: 76px;
      padding: 12px;
      border: 1px solid #eef1f6;
      border-radius: 12px;
      background: #fbfcfe;
      box-shadow: 0 5px 14px rgba(31, 40, 72, 0.06);
    }
    .metric.primary {
      background: var(--accent-soft);
      border-color: #b8e5dc;
    }
    .metric.blue {
      background: var(--blue-soft);
      border-color: #c9d9ff;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
    }
    .metric strong {
      display: block;
      margin-top: 6px;
      font-size: 22px;
      line-height: 1.1;
    }
    .metric em {
      display: block;
      margin-top: 6px;
      color: var(--muted);
      font-size: 11px;
      font-style: normal;
    }
    .exports {
      display: flex;
      gap: 8px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
    }
    .exports a {
      flex: 1;
      text-align: center;
      text-decoration: none;
      color: var(--ink);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 7px 8px;
      font-size: 13px;
      background: #fff;
      transition: transform 160ms ease, box-shadow 160ms ease;
    }
    .exports a:hover {
      transform: translateY(-1px);
      box-shadow: 0 6px 14px rgba(32, 47, 89, 0.12);
    }
    .eval-tools {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      padding: 0 14px 12px;
      border-bottom: 1px solid var(--line);
    }
    .eval-tools a, button {
      cursor: pointer;
      text-align: center;
      text-decoration: none;
      color: var(--ink);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
      background: #fff;
      font: inherit;
      font-size: 13px;
      transition: transform 160ms ease, box-shadow 160ms ease;
    }
    .eval-tools a:hover, button:hover {
      transform: translateY(-1px);
      box-shadow: 0 6px 14px rgba(32, 47, 89, 0.12);
    }
    .results-panel {
      overflow: auto;
    }
    .chart {
      padding: 14px;
      border-bottom: 1px solid var(--line);
    }
    .bar-row {
      display: grid;
      grid-template-columns: 90px minmax(0, 1fr) 70px;
      align-items: center;
      gap: 10px;
      margin: 10px 0;
      color: var(--muted);
      font-size: 12px;
    }
    .track {
      height: 10px;
      overflow: hidden;
      border-radius: 999px;
      background: #edf1f7;
    }
    .fill {
      width: 0%;
      height: 100%;
      border-radius: inherit;
      background: var(--accent);
      transition: width 240ms ease;
    }
    .fill.blue { background: var(--blue); }
    .fill.warn { background: #d97706; }
    .turn-table-wrap {
      padding: 12px 14px 14px;
      border-bottom: 1px solid var(--line);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 12px;
    }
    th {
      color: var(--muted);
      font-weight: 650;
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 7px 6px;
    }
    td {
      border-bottom: 1px solid #eef1f6;
      padding: 8px 6px;
      vertical-align: top;
      word-break: break-word;
    }
    .pill {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      background: #eef1f6;
      color: var(--muted);
      font-size: 11px;
      white-space: nowrap;
    }
    .pill.hit {
      background: var(--warn);
      color: #8a5a00;
    }
    .turn-eval {
      margin-top: 7px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 64px 78px;
      gap: 6px;
    }
    .turn-eval input, .turn-eval select {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 5px;
      padding: 5px 6px;
      font: inherit;
      font-size: 11px;
      background: #fff;
    }
    .report-note {
      margin: 12px 14px 0;
      padding: 10px 12px;
      border: 1px solid #d7eadf;
      border-radius: 12px;
      background: #f2fbf5;
      color: #315c42;
      font-size: 12px;
    }
    .eval-hint {
      margin: 10px 14px 0;
      padding: 10px 12px;
      border: 1px solid #d9dee8;
      border-radius: 12px;
      background: #f8fafc;
      color: var(--muted);
      font-size: 12px;
    }
    .eval-hint strong {
      color: var(--ink);
    }
    .quick-eval {
      margin: 10px 14px 0;
      padding: 12px;
      border: 1px solid #d9dee8;
      border-radius: 14px;
      background: linear-gradient(180deg, #ffffff, #f9fbff);
      box-shadow: 0 10px 22px rgba(28, 39, 76, 0.08);
    }
    .quick-eval-title {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
      font-weight: 650;
    }
    .quick-eval-title small {
      color: var(--muted);
      font-weight: 500;
    }
    .quick-eval-grid {
      display: grid;
      grid-template-columns: 1fr 110px 120px;
      gap: 8px;
    }
    .quick-eval input,
    .quick-eval select,
    .quick-eval button {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px 10px;
      font: inherit;
      font-size: 13px;
      background: #fff;
    }
    .quick-eval button {
      cursor: pointer;
      background: var(--accent-soft);
      border-color: #b8e5dc;
      color: var(--ink);
    }
    .quick-eval button:active {
      transform: translateY(1px);
    }
    .quick-eval-status {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .steps {
      max-height: 220px;
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
    .chart strong {
      font-family: "JetBrains Mono", monospace;
      font-size: 11px;
      color: #2a3450;
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
      <div class="side-head">Results <small>live session</small></div>
      <div class="report-note">
        Report baseline: 32 ms window, 16 ms hop, 4-frame barge-in threshold, theoretical detection latency 64 ms.
      </div>
      <div class="eval-hint">
        <strong>How to evaluate:</strong> speak one turn first, then use the Quick evaluation box or the Turn Results row below to enter the reference transcript, relevance score, context score, and barge-in label.
      </div>
      <div class="quick-eval">
        <div class="quick-eval-title">
          <span>Quick evaluation</span>
          <small id="quick-eval-target">No turn selected yet</small>
        </div>
        <div class="quick-eval-grid">
          <input id="quick-reference" type="text" placeholder="reference transcript for WER">
          <select id="quick-relevance">
            <option value="">NLP relevance</option>
            <option value="1">1/5</option>
            <option value="2">2/5</option>
            <option value="3">3/5</option>
            <option value="4">4/5</option>
            <option value="5">5/5</option>
          </select>
          <select id="quick-context">
            <option value="">context</option>
            <option value="1">1/5</option>
            <option value="2">2/5</option>
            <option value="3">3/5</option>
            <option value="4">4/5</option>
            <option value="5">5/5</option>
          </select>
          <select id="quick-barge">
            <option value="unlabeled">barge-in label</option>
            <option value="intentional">intent</option>
            <option value="false_positive">false</option>
          </select>
        </div>
        <div class="quick-eval-grid" style="margin-top:8px; grid-template-columns: 1fr 1fr;">
          <button id="quick-save">Save to latest turn</button>
          <button id="quick-mark">Mark interruptions intentional</button>
        </div>
        <div id="quick-eval-status" class="quick-eval-status">Speak a turn to enable evaluation.</div>
      </div>
      <div class="metric-grid">
        <div class="metric primary"><span>Avg Response Start</span><strong id="m-response">-</strong><em>speech end to playback</em></div>
        <div class="metric blue"><span>Avg ASR Latency</span><strong id="m-asr">-</strong><em>speech end to transcript</em></div>
        <div class="metric"><span>Total Turns</span><strong id="m-turns">0</strong><em>parsed user turns</em></div>
        <div class="metric"><span>WER / Barge-ins</span><strong id="m-wer">-</strong><em><span id="m-barge">0</span> interruptions</em></div>
      </div>
      <div class="side-head">Average Latency</div>
      <div class="chart">
        <div class="bar-row"><span>ASR</span><div class="track"><div id="bar-asr" class="fill blue"></div></div><strong id="c-asr">-</strong></div>
        <div class="bar-row"><span>LLM token</span><div class="track"><div id="bar-llm" class="fill"></div></div><strong id="c-llm">-</strong></div>
        <div class="bar-row"><span>TTS phrase</span><div class="track"><div id="bar-tts" class="fill"></div></div><strong id="c-tts">-</strong></div>
        <div class="bar-row"><span>Response</span><div class="track"><div id="bar-response" class="fill warn"></div></div><strong id="c-response">-</strong></div>
      </div>
      <div class="side-head">Turn Results</div>
      <div class="turn-table-wrap">
        <table>
          <thead>
            <tr>
              <th style="width:38px">#</th>
              <th>Parsed Message</th>
              <th style="width:70px">ASR</th>
              <th style="width:82px">Response</th>
              <th style="width:70px">State</th>
            </tr>
          </thead>
          <tbody id="turn-rows">
            <tr><td colspan="5" style="color: var(--muted);">No turns yet.</td></tr>
          </tbody>
        </table>
      </div>
      <div class="exports">
        <a href="/api/metrics.json" target="_blank">Download JSON</a>
        <a href="/api/metrics.csv" target="_blank">Download CSV</a>
      </div>
      <div class="eval-tools">
        <button id="mark-intentional">Mark interruptions intentional</button>
        <a href="/api/report.md" target="_blank">Report Markdown</a>
      </div>
      <div class="side-head">Steps</div>
      <div id="steps" class="steps"></div>
    </aside>
  </main>
  <script>
    const messages = document.getElementById("messages");
    const steps = document.getElementById("steps");
    const statusEl = document.getElementById("status");
    const empty = document.getElementById("empty");
    const quickReference = document.getElementById("quick-reference");
    const quickRelevance = document.getElementById("quick-relevance");
    const quickContext = document.getElementById("quick-context");
    const quickBarge = document.getElementById("quick-barge");
    const quickSave = document.getElementById("quick-save");
    const quickMark = document.getElementById("quick-mark");
    const quickStatus = document.getElementById("quick-eval-status");
    const quickTarget = document.getElementById("quick-eval-target");
    const metricEls = {
      turns: document.getElementById("m-turns"),
      barge: document.getElementById("m-barge"),
      wer: document.getElementById("m-wer"),
      asr: document.getElementById("m-asr"),
      response: document.getElementById("m-response"),
      chartAsr: document.getElementById("c-asr"),
      chartLlm: document.getElementById("c-llm"),
      chartTts: document.getElementById("c-tts"),
      chartResponse: document.getElementById("c-response"),
      barAsr: document.getElementById("bar-asr"),
      barLlm: document.getElementById("bar-llm"),
      barTts: document.getElementById("bar-tts"),
      barResponse: document.getElementById("bar-response"),
      rows: document.getElementById("turn-rows")
    };
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
      refreshMetrics();
    }

    function fmtMs(value) {
      if (typeof value !== "number") return "-";
      if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
      return `${Math.round(value)} ms`;
    }

    function barWidth(value, maxValue) {
      if (typeof value !== "number" || !maxValue) return "0%";
      return `${Math.max(4, Math.min(100, (value / maxValue) * 100))}%`;
    }

    function renderTurnRows(turns) {
      if (!turns.length) {
        metricEls.rows.innerHTML = `<tr><td colspan="5" style="color: var(--muted);">No turns yet. Speak one turn, then edit the fields in that row.</td></tr>`;
        return;
      }
      metricEls.rows.innerHTML = turns.slice(-6).reverse().map(turn => `
        <tr>
          <td>${turn.turn_id}</td>
          <td>
            ${escapeHtml(turn.transcript || "-")}
            <div class="turn-eval">
              <input data-turn="${turn.turn_id}" data-field="reference_transcript" value="${escapeAttr(turn.reference_transcript || "")}" placeholder="reference for WER">
              <select data-turn="${turn.turn_id}" data-field="nlp_relevance_score">${scoreOptions(turn.nlp_relevance_score)}</select>
              <select data-turn="${turn.turn_id}" data-field="barge_in_label">${bargeOptions(turn.barge_in_label)}</select>
            </div>
          </td>
          <td>${fmtMs(turn.asr_latency_ms)}</td>
          <td>${fmtMs(turn.response_start_latency_ms)}</td>
          <td><span class="pill ${turn.interrupted ? "hit" : ""}">${turn.interrupted ? "cut off" : "ok"}</span></td>
        </tr>
      `).join("");
      metricEls.rows.querySelectorAll("input, select").forEach(control => {
        control.addEventListener("change", saveEvaluation);
      });
    }

    function escapeAttr(text) {
      return escapeHtml(text).replace(/"/g, "&quot;");
    }

    function scoreOptions(value) {
      const options = [`<option value="">NLP</option>`];
      for (let score = 1; score <= 5; score += 1) {
        options.push(`<option value="${score}" ${Number(value) === score ? "selected" : ""}>${score}/5</option>`);
      }
      return options.join("");
    }

    function bargeOptions(value) {
      return [
        ["unlabeled", "label"],
        ["intentional", "intent"],
        ["false_positive", "false"]
      ].map(([raw, label]) => `<option value="${raw}" ${value === raw ? "selected" : ""}>${label}</option>`).join("");
    }

    function saveEvaluation(event) {
      const turnId = event.target.dataset.turn;
      const row = event.target.closest("tr");
      const payload = {};
      row.querySelectorAll("input, select").forEach(control => {
        payload[control.dataset.field] = control.value;
      });
      fetch(`/api/turns/${turnId}/evaluation`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      }).then(refreshMetrics);
    }

    function getLatestTurnId(turns) {
      if (!Array.isArray(turns) || !turns.length) return null;
      return turns[turns.length - 1].turn_id;
    }

    function saveQuickEvaluation(turnId) {
      if (!turnId) {
        quickStatus.textContent = "Speak a turn first.";
        return;
      }
      const payload = {
        reference_transcript: quickReference.value,
        nlp_relevance_score: quickRelevance.value,
        nlp_context_score: quickContext.value,
        barge_in_label: quickBarge.value
      };
      quickStatus.textContent = `Saving to turn ${turnId}...`;
      fetch(`/api/turns/${turnId}/evaluation`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      }).then(() => {
        quickStatus.textContent = `Saved to turn ${turnId}.`;
        refreshMetrics();
      });
    }

    function refreshMetrics() {
      fetch("/api/metrics")
        .then(response => response.json())
        .then(data => {
          const summary = data.summary;
          const latestTurnId = getLatestTurnId(data.turns || []);
          quickTarget.textContent = latestTurnId ? `Latest turn #${latestTurnId}` : "No turn selected yet";
          quickSave.disabled = !latestTurnId;
          quickMark.disabled = !latestTurnId;
          if (!latestTurnId) {
            quickStatus.textContent = "Speak a turn to enable evaluation.";
          }
          metricEls.turns.textContent = data.session.turns;
          metricEls.barge.textContent = data.session.barge_ins;
          metricEls.wer.textContent = typeof summary.avg_word_error_rate === "number" ? `${(summary.avg_word_error_rate * 100).toFixed(1)}%` : "-";
          metricEls.asr.textContent = fmtMs(summary.avg_asr_latency_ms);
          metricEls.response.textContent = fmtMs(summary.avg_response_start_latency_ms);
          metricEls.chartAsr.textContent = fmtMs(summary.avg_asr_latency_ms);
          metricEls.chartLlm.textContent = fmtMs(summary.avg_llm_first_token_latency_ms);
          metricEls.chartTts.textContent = fmtMs(summary.avg_tts_first_phrase_latency_ms);
          metricEls.chartResponse.textContent = fmtMs(summary.avg_response_start_latency_ms);
          const values = [
            summary.avg_asr_latency_ms,
            summary.avg_llm_first_token_latency_ms,
            summary.avg_tts_first_phrase_latency_ms,
            summary.avg_response_start_latency_ms
          ].filter(value => typeof value === "number");
          const maxValue = Math.max(1000, ...values);
          metricEls.barAsr.style.width = barWidth(summary.avg_asr_latency_ms, maxValue);
          metricEls.barLlm.style.width = barWidth(summary.avg_llm_first_token_latency_ms, maxValue);
          metricEls.barTts.style.width = barWidth(summary.avg_tts_first_phrase_latency_ms, maxValue);
          metricEls.barResponse.style.width = barWidth(summary.avg_response_start_latency_ms, maxValue);
          renderTurnRows(data.turns || []);
        });
    }

    fetch("/api/events")
      .then(response => response.json())
      .then(events => events.forEach(handle));
    refreshMetrics();
    setInterval(refreshMetrics, 3000);

    document.getElementById("mark-intentional").addEventListener("click", () => {
      fetch("/api/metrics/mark-intentional", {method: "POST"}).then(refreshMetrics);
    });
    quickSave.addEventListener("click", () => {
      fetch("/api/metrics").then(response => response.json()).then(data => {
        saveQuickEvaluation(getLatestTurnId(data.turns || []));
      });
    });
    quickMark.addEventListener("click", () => {
      fetch("/api/metrics/mark-intentional", {method: "POST"}).then(() => {
        quickBarge.value = "intentional";
        quickStatus.textContent = "Marked interruptions intentional.";
        refreshMetrics();
      });
    });

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
    metrics = PerformanceMetrics(
        audio_window_ms=32.0,
        audio_hop_ms=16.0,
        barge_in_frames=4,
        sample_rate=16000,
        model_name=config.llm_model,
        asr_model=config.whisper_model,
        tts_backend=config.tts_backend,
    )
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
                metrics=metrics,
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

    @app.get("/api/metrics")
    def metrics_snapshot():
        return jsonify(metrics.snapshot())

    @app.get("/api/metrics.json")
    def metrics_json():
        return jsonify(metrics.snapshot())

    @app.get("/api/metrics.csv")
    def metrics_csv():
        snapshot = metrics.snapshot()
        rows = snapshot["turns"]
        fieldnames = list(rows[0].keys()) if rows else [
            "turn_id",
            "transcript",
            "reference_transcript",
            "word_error_rate",
            "nlp_relevance_score",
            "nlp_context_score",
            "barge_in_label",
            "interrupted",
            "user_speech_duration_ms",
            "asr_latency_ms",
            "llm_first_token_latency_ms",
            "llm_token_count",
            "llm_median_token_latency_ms",
            "llm_token_throughput_tps",
            "tts_first_phrase_latency_ms",
            "response_start_latency_ms",
            "agent_turn_duration_ms",
            "barge_in_after_playback_ms",
            "barge_in_latency_ms",
            "asr_substitutions",
            "asr_insertions",
            "asr_deletions",
            "back_transcription_wer",
            "audio_drop_count",
            "tts_failure_count",
            "first_assistant_phrase",
        ]

        def generate():
            import csv
            import io

            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=fieldnames)
            writer.writeheader()
            yield buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)
            for row in rows:
                writer.writerow(row)
                yield buffer.getvalue()
                buffer.seek(0)
                buffer.truncate(0)

        return Response(
            generate(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=speech_pro_metrics.csv"},
        )

    @app.post("/api/metrics/mark-intentional")
    def mark_intentional():
        metrics.mark_all_interruptions_intentional()
        return jsonify(metrics.snapshot())

    @app.post("/api/turns/<int:turn_id>/evaluation")
    def update_turn_evaluation(turn_id: int):
        payload = request.get_json(silent=True) or {}
        relevance = payload.get("nlp_relevance_score")
        context = payload.get("nlp_context_score")
        row = metrics.update_turn_evaluation(
            turn_id=turn_id,
            reference_transcript=payload.get("reference_transcript"),
            nlp_relevance_score=int(relevance) if str(relevance).isdigit() else None,
            nlp_context_score=int(context) if str(context).isdigit() else None,
            barge_in_label=payload.get("barge_in_label"),
        )
        return jsonify({"ok": row is not None, "turn": row})

    @app.get("/api/report.md")
    def report_markdown():
        snapshot = metrics.snapshot()
        session = snapshot["session"]
        summary = snapshot["summary"]
        report = f"""# Speech Pro Experimental Results

## Speech Layer

| Metric | Result |
|---|---:|
| ASR model | {session["asr_model"]} |
| Sampling rate | {session["sample_rate"]} Hz |
| Window / hop | {session["window_ms"]} ms / {session["hop_ms"]} ms |
| Avg ASR latency | {summary["avg_asr_latency_ms"]} ms |
| Avg WER | {summary["avg_word_error_rate"]} |

## NLP Layer

| Metric | Result |
|---|---:|
| LLM model | {session["llm_model"]} |
| Avg first-token latency | {summary["avg_llm_first_token_latency_ms"]} ms |
| Avg median token latency | {summary.get("avg_llm_median_token_latency_ms")} ms |
| Avg token throughput | {summary.get("avg_llm_token_throughput_tps")} tps |
| Avg relevance score | {summary["avg_nlp_relevance_score"]} / 5 |
| Avg context score | {summary["avg_nlp_context_score"]} / 5 |

## TTS Layer

| Metric | Result |
|---|---:|
| TTS backend | {session["tts_backend"]} |
| Avg first phrase latency | {summary["avg_tts_first_phrase_latency_ms"]} ms |

## Full-Duplex Layer

| Metric | Result |
|---|---:|
| Avg response start latency | {summary["avg_response_start_latency_ms"]} ms |
| Theoretical barge-in latency | {session["theoretical_barge_in_latency_ms"]} ms |
| Barge-in count | {session["barge_ins"]} |
| Intentional barge-in success rate | {summary["intentional_barge_in_success_rate"]} |
| False barge-in rate | {summary["false_barge_in_rate"]} |
| Barge-in latency p50 | {summary.get("barge_in_p50_ms")} ms |
| Barge-in latency p90 | {summary.get("barge_in_p90_ms")} ms |
| Barge-in latency p99 | {summary.get("barge_in_p99_ms")} ms |

## ASR & TTS Diagnostics

| Metric | Result |
|---|---:|
| Total ASR substitutions | {summary.get("total_asr_substitutions")} |
| Total ASR insertions | {summary.get("total_asr_insertions")} |
| Total ASR deletions | {summary.get("total_asr_deletions")} |
| Avg back-transcription WER | {summary.get("avg_back_transcription_wer")} |
| Total audio drop count | {summary.get("total_audio_drop_count")} |
| TTS failure rate (per-turn) | {summary.get("tts_failure_rate")} |
"""
        return Response(report, mimetype="text/markdown")

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
