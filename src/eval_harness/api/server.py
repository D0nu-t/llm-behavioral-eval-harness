"""
api/server.py

Single-process FastAPI server. Runs the eval pipeline in a thread pool and
streams results to the browser over SSE. Serves the dashboard as an inline
HTML response — no Streamlit, no file IPC, no working directory issues.

Launch:
    uvicorn eval_harness.api.server:app --reload --port 8000

Then open http://localhost:8000
"""

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

load_dotenv()

app = FastAPI(title="LLM Behavioral Eval Harness")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=1)
ROOT = Path(__file__).resolve().parents[4]
# DATASET_PATH = ROOT / "datasets" / "sycophancy" / "opinion_assertion.jsonl"
DATASET_PATH = Path("C:/Users/donut/coding_projects/llm-behavioral-eval-harness/datasets/sycophancy/opinion_assertion.jsonl")


def _build_orchestrator():
    from eval_harness.backends.hf_transformer import HFTransformerBackend
    from eval_harness.logging.mlflow_logger import MLflowLogger
    from eval_harness.orchestrator.orchestrator import EvalOrchestrator
    from eval_harness.probes.sycophancy import OpinionAssertionProbe
    from eval_harness.scorers.rubric import SimpleSycophancyScorer
    class HFTransformerBackendWithModelName(HFTransformerBackend):
        def model_name(self) -> str:
            return self.model.config._name_or_path

    return EvalOrchestrator(
        HFTransformerBackendWithModelName=HFTransformerBackendWithModelName(
            model_name=os.getenv("MODEL_NAME", "sshleifer/tiny-gpt2")
        ),
        probe=OpinionAssertionProbe(str(DATASET_PATH)),
        scorer=SimpleSycophancyScorer(),
        logger=MLflowLogger("behavioral_eval"),
    )


def _run_blocking():
    """
    Synchronous generator consumed by the async SSE wrapper.
    Runs in a thread pool so model inference does not block the event loop.
    """
    try:
        orchestrator = _build_orchestrator()
        model_name = os.getenv("MODEL_NAME", "sshleifer/tiny-gpt2")

        for result in orchestrator.run():
            result["model_name"] = model_name
            yield f"data: {json.dumps(result)}\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
    finally:
        yield 'data: {"done": true}\n\n'

async def _sse_generator():
    loop = asyncio.get_event_loop()
    gen = _run_blocking()

    def _next_chunk():
        try:
            return next(gen)
        except StopIteration:
            return None

    while True:
        chunk = await loop.run_in_executor(_executor, _next_chunk)

        if chunk is None:
            break

        yield chunk


@app.get("/run/stream")
async def stream_eval():
    """
    SSE endpoint. Connect with EventSource('/run/stream').
    Emits one JSON object per probe item, then {"done": true}.
    Fields: item_id, score, passed, reasoning, response_text,
            latency_ms, input_tokens, output_tokens, drift_per_layer.
    """
    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "dataset_exists": DATASET_PATH.exists()}


# ---------------------------------------------------------------------------
# Inline dashboard — zero extra dependencies, no separate dev server.
# EventSource connects to /run/stream on the same origin (no CORS needed).
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Language Model Sycophancy Tester</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: ui-monospace, 'Cascadia Code', monospace;
         background: #0f0f0f; color: #e0e0e0; padding: 1.5rem; }
  h1   { font-size: 1rem; font-weight: 500; color: #fff;
         letter-spacing: -.01em; margin-bottom: 1.25rem; }
  .controls { display: flex; gap: 8px; margin-bottom: 1.25rem; }
  button { font-family: inherit; font-size: 12px; padding: 6px 14px;
           background: #1e1e1e; border: 1px solid #333; color: #e0e0e0;
           border-radius: 4px; cursor: pointer; }
  button:hover { background: #2a2a2a; }
  button:disabled { opacity: .4; cursor: not-allowed; }
  .metrics { display: grid; grid-template-columns: repeat(4, 1fr);
             gap: 8px; margin-bottom: 1.25rem; }
  .metric  { background: #1a1a1a; border: 1px solid #222;
             border-radius: 6px; padding: 10px 12px; }
  .metric-label { font-size: 10px; color: #666; text-transform: uppercase;
                  letter-spacing: .06em; margin-bottom: 4px; }
  .metric-val   { font-size: 20px; font-weight: 500; }
  .green { color: #4ade80; }
  .red   { color: #f87171; }
  .charts { display: grid; grid-template-columns: 1fr 1fr;
            gap: 12px; margin-bottom: 1.25rem; }
  .card  { background: #1a1a1a; border: 1px solid #222;
           border-radius: 6px; padding: 12px; }
  .card h2 { font-size: 10px; color: #555; text-transform: uppercase;
             letter-spacing: .06em; margin-bottom: 10px; }
  .table-wrap { background: #1a1a1a; border: 1px solid #222;
                border-radius: 6px; overflow: hidden; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th  { padding: 6px 12px; text-align: left; font-size: 10px; color: #555;
        text-transform: uppercase; letter-spacing: .06em;
        background: #141414; border-bottom: 1px solid #222; }
  td  { padding: 8px 12px; border-bottom: 1px solid #1a1a1a;
        vertical-align: top; }
  tr:last-child td { border-bottom: none; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 3px;
           font-size: 10px; font-weight: 600; letter-spacing: .04em; }
  .pass { background: #052e16; color: #4ade80; }
  .fail { background: #2d0f0f; color: #f87171; }
  .resp { color: #666; font-size: 11px; max-width: 300px;
          white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .status { margin-top: 10px; font-size: 11px; color: #444;
            display: flex; align-items: center; gap: 6px; min-height: 18px; }
  .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
  .pulsing  { animation: pulse 1s infinite; color: #facc15; }
  .dot-done { color: #4ade80; }
  .err { color: #f87171; margin-top: 8px; font-size: 12px; }
</style>
</head>
<body>
<h1>language model sycophancy tester</h1>
<h1>model being tested: <span id="modelName"></span></h1>
<div class="controls">
  <button id="runBtn" onclick="startRun()">▶ run eval</button>
  <button onclick="clearAll()">↺ clear</button>
</div>

<div class="metrics">
  <div class="metric">
    <div class="metric-label">total</div>
    <div class="metric-val" id="mTotal">—</div>
  </div>
  <div class="metric">
    <div class="metric-label">passed</div>
    <div class="metric-val green" id="mPass">—</div>
  </div>
  <div class="metric">
    <div class="metric-label">failed</div>
    <div class="metric-val red" id="mFail">—</div>
  </div>
  <div class="metric">
    <div class="metric-label">mean drift</div>
    <div class="metric-val" id="mDrift">—</div>
  </div>
</div>

<div class="charts">
  <div class="card">
    <h2>score per item</h2>
    <canvas id="scoreChart" height="130"></canvas>
  </div>
  <div class="card">
    <h2>cosine drift per layer — latest item</h2>
    <canvas id="driftChart" height="130"></canvas>
  </div>
</div>

<div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th>item id</th><th>result</th><th>score</th>
        <th>mean drift</th><th>latency</th><th>response</th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
</div>

<div class="status" id="statusBar">
  <span class="dot" id="sDot" style="display:none"></span>
  <span id="sTxt"></span>
</div>
<div class="err" id="errMsg"></div>

<script>
let results = [], es = null;
let scoreChart, driftChart;

const chartDefaults = {
  plugins: { legend: { display: false } },
  animation: false,
  scales: {
    x: { ticks: { color: '#555' }, grid: { color: '#222' } },
    y: { ticks: { color: '#555' }, grid: { color: '#222' } },
  }
};

function initCharts() {
  scoreChart = new Chart(document.getElementById('scoreChart'), {
    type: 'bar',
    data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderRadius: 3 }] },
    options: { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, min: 0, max: 1 } } }
  });

  driftChart = new Chart(document.getElementById('driftChart'), {
    type: 'line',
    data: { labels: [], datasets: [{ data: [],
      borderColor: '#f97316', backgroundColor: 'rgba(249,115,22,.12)',
      fill: true, tension: 0.3, pointRadius: 4 }] },
    options: { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, min: 0 } } }
  });
}

function setStatus(txt, cls) {
  document.getElementById('sDot').style.display = txt ? 'block' : 'none';
  document.getElementById('sDot').className = 'dot ' + (cls || '');
  document.getElementById('sTxt').textContent = txt;
}

function updateMetrics() {
  const n = results.length, p = results.filter(r => r.passed).length;
  const driftVals = results.filter(r => r.drift_per_layer)
    .map(r => r.drift_per_layer.reduce((a,b)=>a+b,0) / r.drift_per_layer.length);
  const meanDrift = driftVals.length
    ? (driftVals.reduce((a,b)=>a+b,0) / driftVals.length).toFixed(4) : '—';
  document.getElementById('mTotal').textContent = n || '—';
  document.getElementById('mPass').textContent  = n ? p : '—';
  document.getElementById('mFail').textContent  = n ? n - p : '—';
  document.getElementById('mDrift').textContent = meanDrift;
}

function updateCharts(r) {
  scoreChart.data.labels.push(r.item_id);
  scoreChart.data.datasets[0].data.push(r.score);
  scoreChart.data.datasets[0].backgroundColor =
    scoreChart.data.datasets[0].data.map(s => s >= 1 ? '#4ade80' : '#f87171');
  scoreChart.update('none');

  if (r.drift_per_layer) {
    driftChart.data.labels = r.drift_per_layer.map((_,i) => `L${i}`);
    driftChart.data.datasets[0].data = r.drift_per_layer;
    driftChart.update('none');
  }
}

function appendRow(r) {
  const drift = r.drift_per_layer
    ? (r.drift_per_layer.reduce((a,b)=>a+b,0)/r.drift_per_layer.length).toFixed(4) : '—';
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td>${r.item_id}</td>
    <td><span class="badge ${r.passed?'pass':'fail'}">${r.passed?'PASS':'FAIL'}</span></td>
    <td>${r.score !== undefined ? r.score.toFixed(1) : '—'}</td>
    <td>${drift}</td>
    <td>${r.latency_ms != null ? r.latency_ms+'ms' : '—'}</td>
    <td class="resp" title="${(r.response_text||'').replace(/"/g,'&quot;')}">${(r.response_text||r.error||'').slice(0,80)}</td>
  `;
  document.getElementById('tbody').appendChild(tr);
}

function clearAll() {
  if (es) { es.close(); es = null; }
  results = [];
  document.getElementById('tbody').innerHTML = '';
  document.getElementById('errMsg').textContent = '';
  scoreChart.data.labels = []; scoreChart.data.datasets[0].data = []; scoreChart.update('none');
  driftChart.data.labels = []; driftChart.data.datasets[0].data = []; driftChart.update('none');
  updateMetrics(); setStatus('');
  document.getElementById('runBtn').disabled = false;
}

function startRun() {
  clearAll();
  document.getElementById('runBtn').disabled = true;
  setStatus('connecting…', 'pulsing');
  es = new EventSource('/run/stream');
  es.onopen = () => setStatus('streaming…', 'pulsing');
  es.onmessage = e => {
    let d; try { d = JSON.parse(e.data); } catch { return; }
    if (d.model_name) {
  document.getElementById('modelName').textContent = d.model_name;
}
    if (d.done) {
      es.close(); es = null;
      document.getElementById('runBtn').disabled = false;
      setStatus(`complete — ${results.length} items`, 'dot-done');
      return;
    }
    if (d.error) {
      document.getElementById('errMsg').textContent = 'Error: ' + d.error;
      es.close(); es = null;
      document.getElementById('runBtn').disabled = false;
      setStatus(''); return;
    }
    results.push(d); appendRow(d); updateMetrics(); updateCharts(d);
    setStatus(`streaming… ${results.length} received`, 'pulsing');
  };
  es.onerror = () => {
    es.close(); es = null;
    document.getElementById('runBtn').disabled = false;
    document.getElementById('errMsg').textContent = 'Connection failed — is the server running?';
    setStatus('');
  };
}

initCharts();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=_DASHBOARD_HTML)
