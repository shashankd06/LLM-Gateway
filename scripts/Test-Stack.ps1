param([switch]$DatabaseOnly)
$ErrorActionPreference = 'Stop'
$projectDirectory = Split-Path $PSScriptRoot -Parent
$workspaceDirectory = Split-Path (Split-Path $projectDirectory -Parent) -Parent
$workspacePython = Join-Path $workspaceDirectory 'work/phase1-py311/Scripts/python.exe'
$projectPython = Join-Path $projectDirectory '.venv/Scripts/python.exe'
if (Test-Path -LiteralPath $projectPython) {
    $pythonExecutable = $projectPython
} elseif (Test-Path -LiteralPath $workspacePython) {
    $pythonExecutable = $workspacePython
} else {
    $pythonExecutable = (Get-Command python -ErrorAction Stop).Source
}
$runnerArguments = @((Join-Path $PSScriptRoot 'verify_stack.py'))
if ($DatabaseOnly) { $runnerArguments += '--database-only' }
& $pythonExecutable @runnerArguments
if ($LASTEXITCODE -ne 0) { throw 'Stack verification failed. See reports/stack-verification.json.' }
