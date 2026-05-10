"""
api/server.py

Single-process FastAPI server. Runs the eval pipeline in a thread pool and
streams results to the browser over SSE. Serves the dashboard as an inline
HTML response.

Launch:
    uvicorn eval_harness.api.server:app --reload --port 8000
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
DATASET_PATH = Path("C:/Users/donut/coding_projects/llm-behavioral-eval-harness/datasets/sycophancy/opinion_assertion.jsonl")


def _build_orchestrator():
    from eval_harness.backends.hf_transformer import HFTransformerBackend
    from eval_harness.logging.mlflow_logger import MLflowLogger
    from eval_harness.orchestrator.orchestrator import EvalOrchestrator
    from eval_harness.probes.sycophancy import OpinionAssertionProbe
    from eval_harness.scorers.rubric import SimpleSycophancyScorer

    return EvalOrchestrator(
        backend=HFTransformerBackend(
            model_name=os.getenv("MODEL_NAME", "sshleifer/tiny-gpt2")
        ),
        probe=OpinionAssertionProbe(str(DATASET_PATH)),
        scorer=SimpleSycophancyScorer(),
        logger=MLflowLogger("behavioral_eval"),
    )


def _run_blocking():
    try:
        orchestrator = _build_orchestrator()
        for result in orchestrator.run():
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
    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "dataset_exists": DATASET_PATH.exists()}


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>LLM Behavioral Eval Harness</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: ui-monospace, 'Cascadia Code', monospace;
         background: #0f0f0f; color: #e0e0e0; padding: 1.5rem; font-size: 13px; }

  /* header */
  h1 { font-size: 1rem; font-weight: 500; color: #fff;
       letter-spacing: -.01em; margin-bottom: .2rem; }
  .subtitle { font-size: 11px; color: #555; margin-bottom: 1.25rem; }

  /* controls */
  .controls { display: flex; gap: 8px; margin-bottom: 1.25rem; align-items: center; }
  button { font-family: inherit; font-size: 12px; padding: 6px 14px;
           background: #1e1e1e; border: 1px solid #333; color: #e0e0e0;
           border-radius: 4px; cursor: pointer; transition: background .15s; }
  button:hover:not(:disabled) { background: #2a2a2a; }
  button:disabled { opacity: .35; cursor: not-allowed; }

  /* summary tiles */
  .tiles { display: grid; grid-template-columns: repeat(6, 1fr);
           gap: 8px; margin-bottom: 1.25rem; }
  .tile { background: #1a1a1a; border: 1px solid #222;
          border-radius: 6px; padding: 10px 12px; }
  .tile-label { font-size: 9px; color: #555; text-transform: uppercase;
                letter-spacing: .07em; margin-bottom: 4px; }
  .tile-val { font-size: 18px; font-weight: 500; }
  .green { color: #4ade80; } .red { color: #f87171; }
  .amber { color: #fbbf24; } .blue { color: #60a5fa; }

  /* chart grid — 2x2 */
  .chart-grid { display: grid; grid-template-columns: 1fr 1fr;
                gap: 12px; margin-bottom: 1.25rem; }
  .card { background: #1a1a1a; border: 1px solid #222;
          border-radius: 6px; padding: 12px; }
  .card-title { font-size: 9px; color: #555; text-transform: uppercase;
                letter-spacing: .07em; margin-bottom: 10px; }

  /* item selector */
  .item-selector { display: flex; align-items: center; gap: 8px;
                   margin-bottom: 8px; }
  .item-selector label { font-size: 10px; color: #555; }
  select { font-family: inherit; font-size: 11px; background: #1a1a1a;
           border: 1px solid #333; color: #e0e0e0; border-radius: 3px;
           padding: 3px 6px; }

  /* table */
  .table-wrap { background: #1a1a1a; border: 1px solid #222;
                border-radius: 6px; overflow: hidden; }
  table { width: 100%; border-collapse: collapse; font-size: 11px; }
  th { padding: 6px 10px; text-align: left; font-size: 9px; color: #555;
       text-transform: uppercase; letter-spacing: .07em;
       background: #141414; border-bottom: 1px solid #222; }
  td { padding: 7px 10px; border-bottom: 1px solid #1a1a1a; vertical-align: top; }
  tr:last-child td { border-bottom: none; }
  tr.selected td { background: #1f2a1f; }
  tr:hover td { background: #1c1c1c; cursor: pointer; }

  .badge { display: inline-block; padding: 1px 7px; border-radius: 3px;
           font-size: 9px; font-weight: 700; letter-spacing: .05em; }
  .pass { background: #052e16; color: #4ade80; }
  .fail { background: #2d0f0f; color: #f87171; }
  .resp { color: #555; max-width: 260px; white-space: nowrap;
          overflow: hidden; text-overflow: ellipsis; }

  /* status */
  .statusbar { margin-top: 10px; font-size: 11px; color: #444;
               display: flex; align-items: center; gap: 6px; min-height: 18px; }
  .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor;
         flex-shrink: 0; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }
  .pulsing  { animation: pulse 1s infinite; color: #facc15; }
  .dot-done { color: #4ade80; }
  .err { color: #f87171; margin-top: 8px; font-size: 11px; }

  /* metric legend pills */
  .legend { display: flex; gap: 12px; margin-bottom: 6px; flex-wrap: wrap; }
  .pill { display: flex; align-items: center; gap: 5px; font-size: 10px; color: #666; }
  .pip { width: 10px; height: 3px; border-radius: 2px; }
</style>
</head>
<body>

<h1>llm behavioral eval harness</h1>
<div class="subtitle">model: <span id="modelName">—</span></div>

<div class="controls">
  <button id="runBtn" onclick="startRun()">&#9654; run eval</button>
  <button onclick="clearAll()">&#8635; clear</button>
</div>

<!-- summary tiles -->
<div class="tiles">
  <div class="tile"><div class="tile-label">total</div><div class="tile-val" id="mTotal">—</div></div>
  <div class="tile"><div class="tile-label">passed</div><div class="tile-val green" id="mPass">—</div></div>
  <div class="tile"><div class="tile-label">failed</div><div class="tile-val red" id="mFail">—</div></div>
  <div class="tile"><div class="tile-label">mean cos drift</div><div class="tile-val amber" id="mDrift">—</div></div>
  <div class="tile"><div class="tile-label">mean norm ratio</div><div class="tile-val blue" id="mNorm">—</div></div>
  <div class="tile"><div class="tile-label">mean eff rank</div><div class="tile-val" id="mRank">—</div></div>
</div>

<!-- item picker for layer charts -->
<div class="item-selector">
  <label>inspect item:</label>
  <select id="itemPicker" onchange="renderLayerCharts()">
    <option value="">— run eval first —</option>
  </select>
</div>

<!-- 2x2 chart grid -->
<div class="chart-grid">
  <div class="card">
    <div class="card-title">score history</div>
    <canvas id="scoreChart" height="140"></canvas>
  </div>
  <div class="card">
    <div class="card-title">cosine drift per layer</div>
    <canvas id="driftChart" height="140"></canvas>
  </div>
  <div class="card">
    <div class="card-title">norm ratio per layer
      <span style="font-size:9px;color:#444;margin-left:6px;">
        &gt;1 = pressure inflated magnitude
      </span>
    </div>
    <canvas id="normChart" height="140"></canvas>
  </div>
  <div class="card">
    <div class="card-title">effective rank per layer
      <span style="font-size:9px;color:#444;margin-left:6px;">
        1.0 = parallel · 2.0 = orthogonal
      </span>
    </div>
    <canvas id="rankChart" height="140"></canvas>
  </div>
</div>

<!-- results table -->
<div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th>item id</th>
        <th>result</th>
        <th>score</th>
        <th>cos drift</th>
        <th>norm ratio</th>
        <th>eff rank</th>
        <th>latency</th>
        <th>response</th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
</div>

<div class="statusbar">
  <span class="dot" id="sDot" style="display:none"></span>
  <span id="sTxt"></span>
</div>
<div class="err" id="errMsg"></div>

<script>
// ── state ──────────────────────────────────────────────────────────────────
let results = [], es = null;
let scoreChart, driftChart, normChart, rankChart;

// ── chart defaults ─────────────────────────────────────────────────────────
const BASE = {
  plugins: { legend: { display: false } },
  animation: false,
  scales: {
    x: { ticks: { color: '#555', font: { size: 10 } }, grid: { color: '#1e1e1e' } },
    y: { ticks: { color: '#555', font: { size: 10 } }, grid: { color: '#1e1e1e' } },
  }
};

function lineChart(id, color, yMin, yMax) {
  return new Chart(document.getElementById(id), {
    type: 'line',
    data: { labels: [], datasets: [{ data: [],
      borderColor: color, backgroundColor: color + '18',
      fill: true, tension: 0.3, pointRadius: 4, pointHoverRadius: 6 }] },
    options: { ...BASE, scales: { ...BASE.scales,
      y: { ...BASE.scales.y,
        ...(yMin !== undefined ? { min: yMin } : {}),
        ...(yMax !== undefined ? { max: yMax } : {}) } } }
  });
}

function initCharts() {
  // score history — bar
  scoreChart = new Chart(document.getElementById('scoreChart'), {
    type: 'bar',
    data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderRadius: 3 }] },
    options: { ...BASE, scales: { ...BASE.scales, y: { ...BASE.scales.y, min: 0, max: 1 } } }
  });

  driftChart = lineChart('driftChart', '#f97316', 0);    // orange
  normChart  = lineChart('normChart',  '#60a5fa', 0);    // blue
  rankChart  = lineChart('rankChart',  '#a78bfa', 1, 2); // purple, range [1,2]
}

// ── status helpers ─────────────────────────────────────────────────────────
function setStatus(txt, cls) {
  const dot = document.getElementById('sDot');
  dot.style.display = txt ? 'block' : 'none';
  dot.className = 'dot ' + (cls || '');
  document.getElementById('sTxt').textContent = txt;
}

function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }
function fmt(v, dp=4) { return v != null ? v.toFixed(dp) : '—'; }

// ── summary tiles ──────────────────────────────────────────────────────────
function updateTiles() {
  const n = results.length, p = results.filter(r => r.passed).length;
  const withDrift = results.filter(r => r.drift_per_layer);
  const withNorm  = results.filter(r => r.norm_ratio_per_layer);
  const withRank  = results.filter(r => r.effective_rank_per_layer);

  document.getElementById('mTotal').textContent = n || '—';
  document.getElementById('mPass').textContent  = n ? p : '—';
  document.getElementById('mFail').textContent  = n ? n - p : '—';
  document.getElementById('mDrift').textContent =
    withDrift.length ? fmt(mean(withDrift.map(r => mean(r.drift_per_layer)))) : '—';
  document.getElementById('mNorm').textContent =
    withNorm.length  ? fmt(mean(withNorm.map(r => mean(r.norm_ratio_per_layer)))) : '—';
  document.getElementById('mRank').textContent =
    withRank.length  ? fmt(mean(withRank.map(r => mean(r.effective_rank_per_layer)))) : '—';
}

// ── score history chart ────────────────────────────────────────────────────
function updateScoreChart(r) {
  scoreChart.data.labels.push(r.item_id);
  scoreChart.data.datasets[0].data.push(r.score ?? 0);
  scoreChart.data.datasets[0].backgroundColor =
    scoreChart.data.datasets[0].data.map(s => s >= 1 ? '#4ade80' : '#f87171');
  scoreChart.update('none');
}

// ── item picker + layer charts ─────────────────────────────────────────────
function addToPicker(r) {
  const sel = document.getElementById('itemPicker');
  if (sel.options[0] && sel.options[0].value === '') sel.remove(0);
  const opt = document.createElement('option');
  opt.value = r.item_id;
  opt.textContent = r.item_id;
  sel.appendChild(opt);
  // auto-select latest
  sel.value = r.item_id;
  renderLayerCharts();
}

function setLayerChart(chart, vals) {
  if (!vals) { chart.data.labels = []; chart.data.datasets[0].data = []; chart.update('none'); return; }
  chart.data.labels = vals.map((_, i) => 'L' + i);
  chart.data.datasets[0].data = vals;
  chart.update('none');
}

function renderLayerCharts() {
  const id = document.getElementById('itemPicker').value;
  const r = results.find(x => x.item_id === id);
  if (!r) return;
  setLayerChart(driftChart, r.drift_per_layer);
  setLayerChart(normChart,  r.norm_ratio_per_layer);
  setLayerChart(rankChart,  r.effective_rank_per_layer);

  // highlight selected row
  document.querySelectorAll('#tbody tr').forEach(tr => {
    tr.classList.toggle('selected', tr.dataset.id === id);
  });
}

// ── table ──────────────────────────────────────────────────────────────────
function appendRow(r) {
  const mDrift = r.drift_per_layer        ? fmt(mean(r.drift_per_layer))        : '—';
  const mNorm  = r.norm_ratio_per_layer   ? fmt(mean(r.norm_ratio_per_layer))   : '—';
  const mRank  = r.effective_rank_per_layer ? fmt(mean(r.effective_rank_per_layer)) : '—';

  const tr = document.createElement('tr');
  tr.dataset.id = r.item_id;
  tr.onclick = () => {
    document.getElementById('itemPicker').value = r.item_id;
    renderLayerCharts();
  };
  tr.innerHTML =
    '<td>' + r.item_id + '</td>' +
    '<td><span class="badge ' + (r.passed ? 'pass' : 'fail') + '">' +
      (r.passed ? 'PASS' : 'FAIL') + '</span></td>' +
    '<td>' + (r.score !== undefined ? r.score.toFixed(1) : '—') + '</td>' +
    '<td>' + mDrift + '</td>' +
    '<td>' + mNorm  + '</td>' +
    '<td>' + mRank  + '</td>' +
    '<td>' + (r.latency_ms != null ? r.latency_ms + 'ms' : '—') + '</td>' +
    '<td class="resp">' + (r.response_text || r.error || '').slice(0, 70) + '</td>';
  document.getElementById('tbody').appendChild(tr);
}

// ── clear ──────────────────────────────────────────────────────────────────
function clearAll() {
  if (es) { es.close(); es = null; }
  results = [];
  document.getElementById('tbody').innerHTML = '';
  document.getElementById('errMsg').textContent = '';
  document.getElementById('modelName').textContent = '—';
  document.getElementById('itemPicker').innerHTML =
    '<option value="">— run eval first —</option>';
  [scoreChart, driftChart, normChart, rankChart].forEach(c => {
    c.data.labels = []; c.data.datasets[0].data = []; c.update('none');
  });
  updateTiles(); setStatus('');
  document.getElementById('runBtn').disabled = false;
}

// ── run ────────────────────────────────────────────────────────────────────
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
      setStatus('complete — ' + results.length + ' items', 'dot-done');
      return;
    }
    if (d.error) {
      document.getElementById('errMsg').textContent = 'Error: ' + d.error;
      es.close(); es = null;
      document.getElementById('runBtn').disabled = false;
      setStatus(''); return;
    }

    results.push(d);
    appendRow(d);
    addToPicker(d);
    updateTiles();
    updateScoreChart(d);
    setStatus('streaming… ' + results.length + ' received', 'pulsing');
  };

  es.onerror = () => {
    es.close(); es = null;
    document.getElementById('runBtn').disabled = false;
    document.getElementById('errMsg').textContent =
      'Connection failed — is the server running?';
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
