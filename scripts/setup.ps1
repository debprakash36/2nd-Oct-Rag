<#
.SYNOPSIS
    Creates the virtual environment and installs dependencies.

.DESCRIPTION
    Equivalent to `make venv`. Kept separate from check.ps1 so the environment is
    built once rather than on every verification run.

    `python` on Windows may resolve to the Microsoft Store alias stub, which
    exits without doing anything. The interpreter is resolved explicitly and the
    failure is reported with what was found, rather than a bare exit code.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $venvPython)) {
    $base = Get-Command python -ErrorAction SilentlyContinue
    if (-not $base) {
        throw "python not found on PATH. Install Python 3.11+ and retry."
    }
    Write-Host "Creating venv with $($base.Source)"
    & $base.Source -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}

& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -e "$root[dev]"

if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

Write-Host "Environment ready. Run scripts/check.ps1 to verify." -ForegroundColor Green