# Workspace-specific helper: models and logs stay outside the Git repository.
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path $PSScriptRoot -Parent
$workspaceDirectory = Split-Path (Split-Path $projectDirectory -Parent) -Parent
$runtimeDirectory = Join-Path $workspaceDirectory 'work/ollama'
$ollamaExecutable = Join-Path $runtimeDirectory 'ollama.exe'
if (-not (Test-Path -LiteralPath $ollamaExecutable)) {
    throw 'Portable Ollama has not finished downloading into work/ollama.'
}
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_MODELS = Join-Path $runtimeDirectory 'models'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$env:OLLAMA_NUM_PARALLEL = '1'
New-Item -ItemType Directory -Path $env:OLLAMA_MODELS -Force | Out-Null
try {
    Invoke-RestMethod 'http://127.0.0.1:11434/api/version' -TimeoutSec 5 | Out-Null
    Write-Host 'Ollama is already running on port 11434.'
} catch {
    Start-Process -FilePath $ollamaExecutable -ArgumentList 'serve' -WindowStyle Hidden `
        -WorkingDirectory $runtimeDirectory `
        -RedirectStandardOutput (Join-Path $runtimeDirectory 'server.stdout.log') `
        -RedirectStandardError (Join-Path $runtimeDirectory 'server.stderr.log') | Out-Null
    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        try {
            Invoke-RestMethod 'http://127.0.0.1:11434/api/version' -TimeoutSec 2 | Out-Null
            $ready = $true
            break
        } catch { Start-Sleep -Seconds 1 }
    }
    if (-not $ready) { throw 'Ollama did not start. Check work/ollama/server.stderr.log.' }
}
foreach ($model in @('qwen2.5:1.5b', 'qwen2.5:7b')) {
    & $ollamaExecutable pull $model
    if ($LASTEXITCODE -ne 0) { throw "Could not download $model." }
}
Write-Host 'Local Ollama and both routing models are ready.'
