# run_server.ps1
# Launch the LLM Behavioral Eval Harness server.
# Run from the project root:
#   .\run_server.ps1
#
# Switch between configs by editing the variables below,
# or override from the command line:
#   $env:MODEL_NAME = "sshleifer/tiny-gpt2"; .\run_server.ps1

param(
    [string]$Mode = "qwen"   # "qwen" or "tiny" (for local CPU testing)
)

if ($Mode -eq "tiny") {
    # ── CPU / tiny-gpt2 (no GPU, no NLA) ────────────────────────────────
    $env:MODEL_NAME = "sshleifer/tiny-gpt2"
    $env:QUANTIZE   = "0"
    $env:DEVICE     = "cpu"
    $env:NLA_MODEL  = ""
    $env:NLA_LAYER  = ""
}
else {
    # ── Qwen2.5-7B 4-bit + NLA (requires CUDA GPU with ~10GB VRAM) ──────
    $env:MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
    $env:QUANTIZE   = "1"
    $env:DEVICE     = "cpu"
    $env:NLA_MODEL  = "kitft/nla-qwen2.5-7b-L20-av"
    $env:NLA_LAYER  = "20"
    $env:NLA_DEVICE = "cpu"
}

Write-Host ""
Write-Host "=== LLM Behavioral Eval Harness ===" -ForegroundColor Cyan
Write-Host "Mode        : $Mode"
Write-Host "Model       : $env:MODEL_NAME"
Write-Host "Device      : $env:DEVICE"
Write-Host "Quantize    : $env:QUANTIZE"
Write-Host "NLA model   : $(if ($env:NLA_MODEL) { $env:NLA_MODEL } else { '(disabled)' })"
Write-Host "NLA layer   : $(if ($env:NLA_LAYER) { $env:NLA_LAYER } else { '(disabled)' })"
Write-Host ""
Write-Host "Dashboard   : http://localhost:8000" -ForegroundColor Green
Write-Host "Health check: http://localhost:8000/health" -ForegroundColor Green
Write-Host ""

uvicorn eval_harness.api.server:app --port 8000
