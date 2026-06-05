from __future__ import annotations

import json
import random
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

from fashion_pipeline import (
  dataset_summary,
  download_dataset,
  load_dataset,
  load_metrics,
  load_model,
  load_training_history,
  predict_sample,
  random_test_example,
  train_with_progress,
)




HOST = "127.0.0.1"
PORT = 8765


def image_payload(image_vector) -> list[float]:
    return np.asarray(image_vector, dtype=np.float32).round(4).tolist()


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.training_running = False
        self.stop_training = False
        self.stage = "idle"
        self.status = "Ready. Download the dataset or start training."
        self.dataset = None
        self.metrics = load_metrics()
        self.history = load_training_history()
        self.batch = None
        self.sample = None
        self.prediction = None
        self.error = None
        self.dataset_path = None
        self.class_names = []
        try:
            self.model_bundle = load_model()
            self.class_names = list(self.model_bundle.get("class_names", []))
            self.dataset_path = self.model_bundle.get("source_path")
            self.stage = "model"
            self.status = "Saved model loaded. Load a test image or retrain."
        except FileNotFoundError:
            self.model_bundle = None

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "trainingRunning": self.training_running,
                "stage": self.stage,
                "status": self.status,
                "dataset": self.dataset,
                "metrics": self.metrics,
                "history": self.history,
                "batch": self.batch,
                "sample": self.sample,
                "prediction": self.prediction,
                "error": self.error,
                "datasetPath": self.dataset_path,
                "classNames": self.class_names,
                "hasModel": self.model_bundle is not None,
            }


STATE = AppState()

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Image Dataset Training Studio</title>
  <style>
    :root {
      --bg: #eef1f5;
      --panel: #ffffff;
      --panel-2: #f7f9fc;
      --text: #111827;
      --muted: #667085;
      --line: #d6dce6;
      --blue: #2563eb;
      --green: #168356;
      --amber: #c27803;
      --red: #c24137;
      --ink: #0f172a;
      --shadow: 0 8px 26px rgba(17, 24, 39, 0.06);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background: var(--bg);
      font-family: "Segoe UI", "Aptos", sans-serif;
    }

    button, input { font: inherit; }

    .app {
      display: grid;
      grid-template-columns: 248px 1fr;
      gap: 10px;
      height: 100vh;
      padding: 10px;
    }

    aside, main, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
    }

    aside {
      display: flex;
      flex-direction: column;
      overflow: auto;
    }

    main {
      display: grid;
      grid-template-rows: 56px 94px minmax(0, 1fr);
      overflow: hidden;
    }

    .brand {
      padding: 16px 16px 12px;
      border-bottom: 1px solid var(--line);
    }

    .brand h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.08;
      letter-spacing: 0;
    }

    .brand p {
      display: none;
      margin: 8px 0 0;
      color: var(--muted);
      line-height: 1.45;
      font-size: 14px;
    }

    .section {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
    }

    .section h2 {
      margin: 0 0 12px;
      font-size: 13px;
      text-transform: uppercase;
      color: var(--muted);
      letter-spacing: 0.08em;
    }

    .actions {
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
    }

    .button {
      height: 38px;
      border: 0;
      color: #fff;
      background: var(--ink);
      cursor: pointer;
      font-weight: 700;
      border-radius: 6px;
    }

    .button.blue { background: var(--blue); }
    .button.green { background: var(--green); }
    .button.light {
      color: var(--text);
      background: #e7ebf2;
    }

    .button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }

    .fields {
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }

    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }

    input {
      width: 100%;
      height: 36px;
      color: var(--text);
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      outline: none;
    }

    input:focus {
      border-color: var(--blue);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.13);
    }

    .hidden-status,
    .status-list { display: none; }

    .status-card strong {
      display: block;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      margin-bottom: 5px;
      letter-spacing: 0.06em;
    }

    .status-card span {
      display: block;
      font-size: 14px;
      line-height: 1.35;
    }

    .topbar {
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    .topbar h2 {
      margin: 0;
      font-size: 20px;
      letter-spacing: 0;
    }

    .topbar p {
      margin: 5px 0 0;
      color: var(--muted);
      font-size: 14px;
    }

    .topbar p { display: none; }

    .flow {
      display: grid;
      grid-template-columns: repeat(8, minmax(92px, 1fr));
      gap: 8px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: #fbfcff;
    }

    .node {
      min-height: 58px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      display: grid;
      place-items: center;
      text-align: center;
      padding: 8px;
      position: relative;
    }

    .node::after {
      content: "";
      width: 18px;
      height: 2px;
      background: var(--line);
      position: absolute;
      right: -15px;
      top: 50%;
    }

    .node:last-child::after { display: none; }

    .node.active {
      border-color: var(--blue);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.13);
      background: #f4f8ff;
    }

    .node .dot {
      width: 11px;
      height: 11px;
      background: var(--ink);
      border-radius: 50%;
      margin-bottom: 7px;
      opacity: 0.18;
    }

    .node.active .dot { opacity: 1; background: var(--blue); }

    .node b {
      font-size: 12px;
      line-height: 1.2;
    }

    .workspace {
      display: grid;
      grid-template-columns: minmax(560px, 1fr) 360px;
      grid-template-rows: minmax(0, 1fr) 285px;
      grid-template-areas:
        "batch sample"
        "chart chart";
      gap: 10px;
      padding: 10px;
      min-height: 0;
      overflow: hidden;
    }

    .batch-panel { grid-area: batch; }
    .sample-panel { grid-area: sample; }
    .chart-panel { grid-area: chart; }

    .panel {
      display: flex;
      flex-direction: column;
      min-height: 0;
      overflow: hidden;
    }

    .panel-head {
      padding: 10px 12px 8px;
      border-bottom: 1px solid var(--line);
    }

    .panel-head h3 {
      margin: 0;
      font-size: 15px;
    }

    .panel-head p {
      display: none;
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
    }

    .chart-panel .panel-head {
      display: grid;
      grid-template-columns: 140px minmax(0, 1fr);
      gap: 10px;
      align-items: center;
      padding: 9px 10px;
      min-height: 66px;
      overflow: hidden;
    }

    .chart-metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      min-width: 0;
      overflow: hidden;
    }

    .metric-pill {
      min-width: 0;
      height: 46px;
      padding: 7px 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8fafc;
      overflow: hidden;
      display: grid;
      align-content: center;
      gap: 3px;
    }

    .metric-pill strong,
    .metric-pill span {
      display: block;
      min-width: 0;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .metric-pill strong {
      color: var(--muted);
      font-size: 9px;
      line-height: 1;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .metric-pill span {
      color: var(--text);
      font-size: 13px;
      font-weight: 800;
      line-height: 1.1;
    }

    .chart-wrap {
      flex: 1;
      min-height: 0;
      padding: 8px;
    }

    #chartCanvas {
      width: 100%;
      height: 100%;
      min-height: 160px;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
    }

    .batch-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      padding: 10px;
      overflow: auto;
      align-content: start;
    }

    .summary-host {
      padding: 10px;
      overflow: auto;
      min-height: 0;
    }

    .image-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 10px;
      display: grid;
      gap: 8px;
      justify-items: center;
      min-height: 184px;
    }

    .image-card.primary {
      border-color: var(--blue);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
    }

    .image-card canvas {
      width: 132px;
      height: 132px;
      image-rendering: pixelated;
      background: #111;
      border-radius: 4px;
    }

    .image-card span {
      font-size: 13px;
      font-weight: 800;
      text-align: center;
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
      padding: 12px;
      width: 100%;
      align-content: start;
      overflow: auto;
    }

    .summary-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 14px;
      min-height: 92px;
      min-width: 0;
      display: grid;
      align-content: center;
      gap: 8px;
      overflow: hidden;
    }

    .summary-card.wide {
      grid-column: 1 / -1;
      min-height: 76px;
    }

    .summary-card strong {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      line-height: 1.2;
      white-space: normal;
    }

    .summary-card span {
      color: var(--text);
      font-size: clamp(17px, 2.1vw, 22px);
      font-weight: 800;
      line-height: 1.15;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .summary-card.wide span {
      font-size: 15px;
      font-weight: 700;
      color: var(--text);
    }

    .sample-area {
      display: grid;
      grid-template-rows: auto 1fr;
      gap: 10px;
      padding: 10px;
      min-height: 0;
      overflow: auto;
    }

    .sample-image {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
      display: grid;
      justify-items: center;
      gap: 8px;
    }

    .sample-image canvas {
      width: 140px;
      height: 140px;
      image-rendering: pixelated;
      background: #111;
      border-radius: 4px;
    }

    .sample-image span {
      font-size: 13px;
      font-weight: 800;
      text-align: center;
    }

    .prob-list {
      display: grid;
      align-content: start;
      gap: 9px;
      padding-top: 2px;
      overflow: visible;
    }

    .bar-row {
      display: grid;
      grid-template-columns: 86px 1fr 44px;
      gap: 6px;
      align-items: center;
      font-size: 12px;
      font-weight: 700;
    }

    .bar {
      height: 15px;
      background: #e6ebf2;
      border-radius: 999px;
      overflow: hidden;
    }

    .bar-fill {
      height: 100%;
      background: var(--blue);
      border-radius: inherit;
    }

    .empty {
      height: 100%;
      display: grid;
      place-items: center;
      color: var(--muted);
      font-size: 14px;
      text-align: center;
      padding: 22px;
    }

    @media (max-width: 1180px) {
      .workspace {
        grid-template-columns: 1fr;
        grid-template-rows: auto auto 285px;
        grid-template-areas:
          "batch"
          "sample"
          "chart";
        overflow: visible;
      }
      .batch-grid { grid-template-columns: repeat(2, 1fr); }
      .flow { grid-template-columns: repeat(4, 1fr); }
      .node::after { display: none; }
      main { overflow: auto; }
    }

    @media (max-width: 820px) {
      .app { grid-template-columns: 1fr; height: auto; }
      aside { width: auto; }
      main { min-height: 900px; }
      .sample-area { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="brand">
        <h1>Image Dataset Training Studio</h1>
        <p>Live Petri-net simulation for dataset loading, batch training, model readiness, and image classification.</p>
      </div>

      <section class="section">
        <h2>Controls</h2>
        <div class="actions">
          <button id="downloadBtn" class="button blue">Use Fashion-MNIST Demo</button>
          <button id="loadFolderBtn" class="button light">Load Folder Dataset</button>
          <button id="trainBtn" class="button green">Start Training</button>
          <button id="stopBtn" class="button" style="background: var(--red); display: none;">Stop Training</button>
          <button id="sampleBtn" class="button light">Load Random Test Image</button>
          <button id="predictBtn" class="button light">Predict Sample</button>
        </div>
      </section>

      <section class="section">
        <h2>Dataset and Training</h2>
        <div class="fields">
          <label>Dataset Folder <input id="datasetPath" type="text" placeholder="data/custom/flowers" /></label>
          <label>Epochs <input id="epochs" type="number" min="1" value="5" /></label>
          <label>Batch Size <input id="batchSize" type="number" min="1" value="512" /></label>
          <label>Delay ms <input id="delayMs" type="number" min="0" value="180" /></label>
        </div>
      </section>

      <section class="section hidden-status">
        <h2>Status</h2>
        <div class="status-list">
          <div class="status-card"><strong>Run</strong><span id="statusText">Loading...</span></div>
          <div class="status-card"><strong>Dataset</strong><span id="datasetText">No dataset loaded.</span></div>
          <div class="status-card"><strong>Training</strong><span id="trainingText">Idle.</span></div>
          <div class="status-card"><strong>Metrics</strong><span id="metricsText">No metrics yet.</span></div>
          <div class="status-card"><strong>Sample</strong><span id="sampleText">No sample loaded.</span></div>
          <div class="status-card"><strong>Prediction</strong><span id="predictionText">No prediction yet.</span></div>
        </div>
      </section>
    </aside>

    <main>
      <div class="topbar">
        <h2>Training Flow and Live Image Batches</h2>
        <p>The batch viewer shows the images currently being used by training. The chart updates during mini-batches.</p>
      </div>

      <div id="flow" class="flow"></div>

      <div class="workspace">
        <section class="panel batch-panel">
          <div class="panel-head">
            <h3 id="batchTitle">Current Training Batch</h3>
            <p id="batchSubtext">Start training to see live batch images.</p>
          </div>
          <div id="batchGrid" class="batch-grid"></div>
        </section>

        <section class="panel sample-panel">
          <div class="panel-head">
            <h3>Sample and Prediction</h3>
          </div>
          <div class="sample-area">
            <div id="sampleImage" class="sample-image"></div>
            <div id="probList" class="prob-list"></div>
          </div>
        </section>

        <section class="panel chart-panel">
          <div class="panel-head">
            <h3>Training Curves</h3>
            <div id="chartStats" class="chart-metrics"></div>
          </div>
          <div class="chart-wrap">
            <canvas id="chartCanvas" width="760" height="220"></canvas>
          </div>
        </section>
      </div>
    </main>
  </div>

  <script>
    const stages = [
      ["idle", "Idle"],
      ["dataset", "Dataset Ready"],
      ["training", "Training"],
      ["model", "Model Ready"],
      ["sample", "Sample Loaded"],
      ["predicting", "Predicting"],
      ["classified", "Classified"],
      ["result", "Result"],
    ];

    const state = { lastBatchKey: "", lastSampleKey: "", stageOverride: null, stageOverrideUntil: 0 };

    function $(id) { return document.getElementById(id); }

    async function postJSON(path, body = {}) {
      const res = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Request failed");
      return data;
    }

    function drawImage(canvas, pixels) {
      const ctx = canvas.getContext("2d");
      const width = 100;
      const height = 100;
      const image = ctx.createImageData(width, height);
      for (let i = 0; i < (width*height); i++) {
        const value = Math.max(0, Math.min(255, Math.round(pixels[i] * 255)));
        const j = i * 4;
        image.data[j] = value;
        image.data[j + 1] = value;
        image.data[j + 2] = value;
        image.data[j + 3] = 255;
      }
      ctx.imageSmoothingEnabled = false;
      ctx.putImageData(image, 0, 0);
    }

    function renderFlow(stage) {
      $("flow").innerHTML = stages.map(([key, label]) => `
        <div class="node ${key === stage ? "active" : ""}">
          <div><div class="dot"></div><b>${label}</b></div>
        </div>
      `).join("");
    }

    function renderBatch(data) {
      const grid = $("batchGrid");
      const batch = data.batch;

      if (!data.trainingRunning) {
        state.lastBatchKey = "";
        grid.className = data.hasModel ? "summary-host" : "batch-grid";
        $("batchTitle").textContent = data.hasModel ? "Model Ready" : "Training Preview";
        $("batchSubtext").textContent = data.hasModel
          ? "Training has finished. Use the prediction controls to inspect model behavior."
          : "Start training to see live batch images.";
        grid.innerHTML = renderModelReadyPanel(data);
        return;
      }

      $("batchTitle").textContent = "Current Training Batch";
      grid.className = "batch-grid";

      if (!batch || !batch.images || batch.images.length === 0) {
        state.lastBatchKey = "";
        grid.innerHTML = `<div class="empty">Preparing the first visible mini-batch.</div>`;
        $("batchSubtext").textContent = "Training is starting...";
        return;
      }

      const key = `${batch.epoch}-${batch.batch}`;
      $("batchSubtext").textContent = `Epoch ${batch.epoch}, batch ${batch.batch} of ${batch.num_batches} - loss ${batch.loss}, batch accuracy ${batch.batch_accuracy}`;
      if (state.lastBatchKey === key) return;
      state.lastBatchKey = key;

      grid.innerHTML = "";
      batch.images.slice(0, 8).forEach((pixels, index) => {
        const card = document.createElement("div");
        card.className = `image-card ${index === 0 ? "primary" : ""}`;
        const canvas = document.createElement("canvas");
        canvas.width = 28;
        canvas.height = 28;
        const label = document.createElement("span");
        label.textContent = batch.labels[index] ?? "Image";
        card.append(canvas, label);
        grid.append(card);
        drawImage(canvas, pixels);
      });
    }

    function renderModelReadyPanel(data) {
      if (!data.hasModel) {
        return `<div class="empty">Start training to see the live mini-batches used by the model.</div>`;
      }

      const metrics = data.metrics || {};
      const dataset = data.dataset || {};
      const history = data.history || [];
      const latest = history.length ? history[history.length - 1] : null;
      const testAccuracy = metrics.test_accuracy ?? "n/a";
      const trainAccuracy = metrics.train_accuracy ?? "n/a";
      const loss = metrics.loss ?? (latest ? latest.loss : "n/a");
      const epochs = metrics.epochs ?? (latest ? latest.epoch : "n/a");
      const trainRows = metrics.train_rows ?? dataset.train_rows ?? "n/a";
      const testRows = metrics.test_rows ?? dataset.test_rows ?? "n/a";
      const datasetName = metrics.dataset_name ?? dataset.name ?? "Selected dataset";
      const classCount = (metrics.class_names || dataset.class_names || []).length || "n/a";

      return `
        <div class="summary-grid">
          <div class="summary-card"><strong>Test Accuracy</strong><span>${testAccuracy}</span></div>
          <div class="summary-card"><strong>Train Accuracy</strong><span>${trainAccuracy}</span></div>
          <div class="summary-card"><strong>Loss</strong><span>${loss}</span></div>
          <div class="summary-card"><strong>Epochs</strong><span>${epochs}</span></div>
          <div class="summary-card wide"><strong>Dataset</strong><span>${datasetName}: ${trainRows} train images, ${testRows} test images, ${classCount} classes</span></div>
          <div class="summary-card wide"><strong>Next Step</strong><span>Load a random test image, then run prediction.</span></div>
        </div>
      `;
    }

    function renderSample(sample, prediction) {
      const box = $("sampleImage");
      if (!sample) {
        state.lastSampleKey = "";
        box.innerHTML = `<div class="empty">Load a test image.</div>`;
        $("probList").innerHTML = `<div class="empty">Run prediction to see probabilities.</div>`;
        return;
      }

      const key = `${sample.index}-${prediction ? prediction.label : ""}`;
      if (state.lastSampleKey !== key) {
        state.lastSampleKey = key;
        box.innerHTML = "";
        const canvas = document.createElement("canvas");
        canvas.width = 28;
        canvas.height = 28;
        const label = document.createElement("span");
        label.textContent = `Actual: ${sample.label}`;
        box.append(canvas, label);
        drawImage(canvas, sample.image);
      }

      if (!prediction) {
        $("probList").innerHTML = `<div class="empty">Run prediction to see class scores.</div>`;
        return;
      }

      const rows = Object.entries(prediction.all_probabilities)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 6)
        .map(([label, value], index) => `
          <div class="bar-row">
            <span>${label}</span>
            <div class="bar"><div class="bar-fill" style="width:${value * 100}%; background:${index === 0 ? "var(--green)" : "var(--blue)"}"></div></div>
            <span>${value.toFixed(3)}</span>
          </div>
        `).join("");
      $("probList").innerHTML = rows;
    }

    function renderChartStats(items) {
      $("chartStats").innerHTML = items.map(item => `
        <div class="metric-pill">
          <strong>${item.label}</strong>
          <span>${item.value}</span>
        </div>
      `).join("");
    }

    function renderChart(history) {
      const canvas = $("chartCanvas");
      const ctx = canvas.getContext("2d");
      const bounds = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      const w = Math.max(320, Math.floor(bounds.width));
      const h = Math.max(160, Math.floor(bounds.height));
      if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
        canvas.width = Math.floor(w * dpr);
        canvas.height = Math.floor(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);

      const left = 52, top = 22, right = w - 18, bottom = h - 36;
      ctx.strokeStyle = "#d6dce6";
      ctx.lineWidth = 1;
      ctx.strokeRect(left, top, right - left, bottom - top);

      ctx.fillStyle = "#667085";
      ctx.font = "12px Segoe UI";
      [0, .25, .5, .75, 1].forEach(v => {
        const y = bottom - (bottom - top) * v;
        ctx.strokeStyle = "#edf1f6";
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(right, y);
        ctx.stroke();
        ctx.fillText(v.toFixed(2), 12, y + 4);
      });

      if (!history || history.length === 0) {
        renderChartStats([
          { label: "Loss", value: "--" },
          { label: "Batch Acc", value: "--" },
          { label: "Test Acc", value: "--" },
          { label: "Progress", value: "Waiting" },
        ]);
        ctx.fillStyle = "#667085";
        ctx.font = "15px Segoe UI";
        ctx.fillText("Waiting for training data.", left + 50, top + 75);
        return;
      }

      const points = history
        .map((p, index) => ({ ...p, x: p.progress ?? p.epoch ?? index + 1 }))
        .filter(p => Number.isFinite(p.x));
      const maxX = Math.max(...points.map(p => p.x));
      const minX = Math.min(...points.map(p => p.x));
      const lossValues = points.map(p => p.loss).filter(Number.isFinite);
      const maxLoss = Math.max(1, ...lossValues);

      function point(xValue, value, ceiling = 1) {
        const x = left + (right - left) * ((xValue - minX) / Math.max(maxX - minX, 1));
        const y = bottom - (bottom - top) * (value / ceiling);
        return [x, y];
      }

      function line(keys, color, ceiling = 1) {
        const usable = points
          .map(p => {
            const key = keys.find(name => Number.isFinite(p[name]));
            return key ? { x: p.x, value: p[key] } : null;
          })
          .filter(Boolean);
        if (usable.length === 0) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.beginPath();
        usable.forEach((p, index) => {
          const [x, y] = point(p.x, p.value, ceiling);
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }

      line(["batch_accuracy", "train_accuracy"], "#2563eb");
      line(["test_accuracy"], "#168356");
      line(["loss"], "#c24137", maxLoss);

      ctx.fillStyle = "#2563eb";
      ctx.fillRect(left, h - 19, 10, 10);
      ctx.fillStyle = "#168356";
      ctx.fillRect(left + 92, h - 19, 10, 10);
      ctx.fillStyle = "#c24137";
      ctx.fillRect(left + 182, h - 19, 10, 10);
      ctx.fillStyle = "#111827";
      ctx.font = "12px Segoe UI";
      ctx.fillText("batch acc", left + 15, h - 10);
      ctx.fillText("test acc", left + 107, h - 10);
      ctx.fillText("loss", left + 197, h - 10);

      const latest = points[points.length - 1];
      const liveAcc = latest.batch_accuracy ?? latest.train_accuracy ?? "n/a";
      const testAcc = latest.test_accuracy ?? "pending";
      renderChartStats([
        { label: "Loss", value: latest.loss ?? "n/a" },
        { label: "Batch Acc", value: liveAcc },
        { label: "Test Acc", value: testAcc },
        { label: "Progress", value: `E${latest.epoch} B${latest.batch ?? "-"}` },
      ]);
    }

    function renderStatus(data) {
      $("statusText").textContent = data.status || "Idle.";
      $("datasetText").textContent = data.dataset
        ? `${data.dataset.name}: ${data.dataset.train_rows} training images, ${data.dataset.test_rows} test images, ${data.dataset.class_names.length} classes.`
        : "Dataset has not been summarized yet.";
      $("trainingText").textContent = data.batch
        ? `Epoch ${data.batch.epoch}, batch ${data.batch.batch} of ${data.batch.num_batches}.`
        : data.trainingRunning ? "Training is starting..." : "Idle.";
      $("metricsText").textContent = data.metrics
        ? `Test ${data.metrics.test_accuracy}, train ${data.metrics.train_accuracy}, ${data.metrics.epochs} epochs.`
        : "No metrics yet.";
      $("sampleText").textContent = data.sample
        ? `Image #${data.sample.index}, actual class ${data.sample.label}.`
        : "No sample loaded.";
      $("predictionText").textContent = data.prediction
        ? `${data.prediction.label}, confidence ${data.prediction.probability}.`
        : "No prediction yet.";

      $("trainBtn").disabled = data.trainingRunning;
      $("downloadBtn").disabled = data.trainingRunning;
      $("sampleBtn").disabled = data.trainingRunning;
      $("predictBtn").disabled = data.trainingRunning || !data.sample;
      $("stopBtn").style.display = data.trainingRunning ? 'block' : 'none';
      if (data.datasetPath && !$("datasetPath").value) $("datasetPath").value = data.datasetPath;
    }

    async function refresh() {
      const data = await fetch("/api/state").then(r => r.json());
      renderStatus(data);
      const stage = state.stageOverride && Date.now() < state.stageOverrideUntil ? state.stageOverride : data.stage;
      renderFlow(stage);
      renderBatch(data);
      renderSample(data.sample, data.prediction);
      renderChart(data.history || []);
    }

    $("downloadBtn").addEventListener("click", async () => {
      $("datasetPath").value = "";
      $("statusText").textContent = "Preparing demo dataset...";
      try { await postJSON("/api/download"); await refresh(); }
      catch (err) { alert(err.message); }
    });

    $("loadFolderBtn").addEventListener("click", async () => {
      try {
        const datasetPath = $("datasetPath").value.trim();
        if (!datasetPath) throw new Error("Enter a dataset folder path first.");
        await postJSON("/api/load_dataset", { datasetPath });
        await refresh();
      }
      catch (err) { alert(err.message); }
    });

    $("trainBtn").addEventListener("click", async () => {
      const body = {
        epochs: Number($("epochs").value || 5),
        batchSize: Number($("batchSize").value || 512),
        delayMs: Number($("delayMs").value || 0),
        datasetPath: $("datasetPath").value.trim(),
      };
      try { await postJSON("/api/train", body); await refresh(); }
      catch (err) { alert(err.message); }
    });

    $("sampleBtn").addEventListener("click", async () => {
      try { await postJSON("/api/sample"); await refresh(); }
      catch (err) { alert(err.message); }
    });

    $("predictBtn").addEventListener("click", async () => {
      try {
        state.stageOverride = "predicting";
        state.stageOverrideUntil = Date.now() + 900;
        renderFlow("predicting");
        $("statusText").textContent = "Prediction running...";
        const data = await postJSON("/api/predict");
        state.stageOverride = "classified";
        state.stageOverrideUntil = Date.now() + 750;
        renderStatus(data);
        renderSample(data.sample, data.prediction);
        renderFlow("classified");
        setTimeout(() => {
          state.stageOverride = null;
          renderFlow("result");
          refresh();
        }, 750);
      }
      catch (err) { alert(err.message); }
    });

    $("stopBtn").addEventListener("click", async () => {
      try {
        $("statusText").textContent = "Stopping training...";
        await postJSON("/api/stop");
        await refresh();
      }
      catch (err) { alert(err.message); }
    });

    refresh();
    setInterval(refresh, 650);
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send(self, status: int, payload: object, content_type: str = "application/json") -> None:
        if content_type == "application/json":
            body = json.dumps(payload).encode("utf-8")
        else:
            body = str(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._send(status, {"error": message})

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, HTML, "text/html")
            return
        if path == "/api/state":
            self._send(200, STATE.snapshot())
            return
        self._error(404, "Not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/download":
                self.handle_download()
            elif path == "/api/load_dataset":
                self.handle_load_dataset()
            elif path == "/api/train":
                self.handle_train()
            elif path == "/api/sample":
                self.handle_sample()
            elif path == "/api/predict":
                self.handle_predict()
            elif path == "/api/stop":
                self.handle_stop()
            else:
                self._error(404, "Not found")
        except Exception as error:
            self._error(500, str(error))

    def handle_download(self) -> None:
        download_dataset()
        summary = dataset_summary(load_dataset())
        with STATE.lock:
            STATE.dataset = summary
            STATE.dataset_path = None
            STATE.class_names = list(summary["class_names"])
            STATE.model_bundle = None
            STATE.metrics = None
            STATE.history = []
            STATE.batch = None
            STATE.sample = None
            STATE.prediction = None
            STATE.stage = "dataset"
            STATE.status = "Fashion-MNIST demo dataset is ready."
            STATE.error = None
        self._send(200, STATE.snapshot())

    def handle_load_dataset(self) -> None:
        body = self._read_json()
        dataset_path = str(body.get("datasetPath", "")).strip()
        if not dataset_path:
            self._error(400, "Enter a dataset folder path.")
            return
        data = load_dataset(dataset_path)
        summary = dataset_summary(data)
        with STATE.lock:
            STATE.dataset = summary
            STATE.dataset_path = summary["source_path"]
            STATE.class_names = list(summary["class_names"])
            STATE.model_bundle = None
            STATE.metrics = None
            STATE.history = []
            STATE.batch = None
            STATE.sample = None
            STATE.prediction = None
            STATE.stage = "dataset"
            STATE.status = f"{summary['name']} is ready."
            STATE.error = None
        self._send(200, STATE.snapshot())

    def handle_train(self) -> None:
        body = self._read_json()
        epochs = int(body.get("epochs", 5))
        batch_size = int(body.get("batchSize", 512))
        delay_ms = int(body.get("delayMs", 180))
        dataset_path = str(body.get("datasetPath", "")).strip() or None

        with STATE.lock:
            if STATE.training_running:
                self._error(409, "Training is already running.")
                return
            STATE.training_running = True
            STATE.stop_training = False
            STATE.dataset_path = dataset_path
            STATE.stage = "training"
            STATE.status = "Training started."
            STATE.history = []
            STATE.batch = None
            STATE.prediction = None
            STATE.error = None

        thread = threading.Thread(
            target=training_worker,
            args=(epochs, batch_size, delay_ms, dataset_path),
            daemon=True,
        )
        thread.start()
        self._send(202, STATE.snapshot())

    def handle_sample(self) -> None:
        with STATE.lock:
            dataset_path = STATE.dataset_path
        sample = random_test_example(random.randint(0, 10_000_000), dataset_path=dataset_path)
        payload = {
            "index": sample["index"],
            "label": sample["label"],
            "label_index": sample["label_index"],
            "image": image_payload(sample["image"]),
            "dataset_name": sample["dataset_name"],
        }
        with STATE.lock:
            STATE.sample = payload
            STATE.prediction = None
            STATE.stage = "sample"
            STATE.status = "Test image loaded."
        self._send(200, STATE.snapshot())

    def handle_predict(self) -> None:
        with STATE.lock:
            sample = STATE.sample
            model_bundle = STATE.model_bundle
        if sample is None:
            self._error(400, "Load a sample first.")
            return
        if model_bundle is None:
            model_bundle = load_model()

        with STATE.lock:
            STATE.stage = "predicting"
            STATE.status = "Prediction running."

        time.sleep(0.35)
        prediction = predict_sample(model_bundle, np.asarray(sample["image"], dtype=np.float32))
        with STATE.lock:
            STATE.model_bundle = model_bundle
            STATE.prediction = prediction
            STATE.stage = "classified"
            STATE.status = "Sample classified."
        time.sleep(0.45)
        with STATE.lock:
            STATE.stage = "result"
            STATE.status = "Prediction complete."
        self._send(200, STATE.snapshot())

    def handle_stop(self) -> None:
        with STATE.lock:
            if not STATE.training_running:
                self._error(400, "Training is not running.")
                return
            STATE.stop_training = True
            STATE.status = "Stopping training..."
        self._send(200, STATE.snapshot())


def training_worker(
    epochs: int,
    batch_size: int,
    delay_ms: int,
    dataset_path: str | None,
) -> None:
    def progress(event: dict[str, object]) -> None:
        with STATE.lock:
            if event["type"] == "batch":
                class_names = list(event.get("class_names") or STATE.class_names)
                labels = [class_names[int(label)] for label in event["preview_labels"]]
                STATE.class_names = class_names
                STATE.batch = {
                    "epoch": event["epoch"],
                    "batch": event["batch"],
                    "num_batches": event["num_batches"],
                    "loss": event["loss"],
                    "batch_accuracy": event["batch_accuracy"],
                    "images": [image_payload(image) for image in event["preview_images"][:8]],
                    "labels": labels,
                }
                STATE.history.append(
                    {
                        "epoch": event["epoch"],
                        "batch": event["batch"],
                        "progress": event["progress"],
                        "loss": event["loss"],
                        "batch_accuracy": event["batch_accuracy"],
                    }
                )
                STATE.stage = "training"
                STATE.status = (
                    f"Training epoch {event['epoch']}, batch {event['batch']} of {event['num_batches']} "
                    f"- loss {event['loss']}, batch accuracy {event['batch_accuracy']}."
                )
            elif event["type"] == "epoch":
                STATE.history.append(
                    {
                        "epoch": event["epoch"],
                        "loss": event["loss"],
                        "train_accuracy": event["train_accuracy"],
                        "test_accuracy": event["test_accuracy"],
                    }
                )
                STATE.status = f"Epoch {event['epoch']} complete."

    try:
        result = train_with_progress(
            epochs=epochs,
            batch_size=batch_size,
            progress_delay_ms=delay_ms,
            progress_callback=progress,
            dataset_path=dataset_path,
            should_stop=lambda: STATE.stop_training,
        )
        with STATE.lock:
            STATE.model_bundle = result["model_bundle"]
            STATE.metrics = result["metrics"]
            STATE.dataset = result["summary"]
            STATE.dataset_path = result["summary"]["source_path"] if dataset_path else None
            STATE.class_names = list(result["summary"]["class_names"])
            STATE.history = result["history"]
            STATE.batch = None
            STATE.training_running = False
            STATE.stop_training = False
            STATE.stage = "model"
            STATE.status = "Training complete. Model is ready."
            STATE.error = None
    except Exception as error:
        with STATE.lock:
            STATE.training_running = False
            STATE.stop_training = False
            STATE.stage = "idle"
            STATE.status = "Training stopped." if error.__class__.__name__ == "KeyboardInterrupt" else "Training failed."
            STATE.error = str(error)
