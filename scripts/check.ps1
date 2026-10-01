<#
.SYNOPSIS
    Runs the full check suite: lint, typecheck, tests.

.DESCRIPTION
    `make check` is the documented entry point, but GNU make is not present on
    every Windows host and installing it is an extra prerequisite before any
    verification can run. This script does the same three steps with the
    project venv and no build tooling.

    Each stage runs even if an earlier one failed, so a single invocation
    reports everything wrong rather than only the first thing.
#>
[CmdletBinding()]
param(
    [switch]$SkipLint,
    [switch]$SkipTypecheck,
    [switch]$SkipTests
)

$ErrorActionPreference = 'Continue'

$python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    Write-Error "virtual environment not found at $python. Run scripts/setup.ps1 first."
    exit 1
}

$failed = @()

function Invoke-Step {
    param([string]$Name, [scriptblock]$Action)

    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        $script:failed += $Name
    }
}

if (-not $SkipLint) {
    Invoke-Step 'ruff check' { & $python -m ruff check app tests scripts alembic }
}

if (-not $SkipTypecheck) {
    Invoke-Step 'mypy' { & $python -m mypy app }
}

if (-not $SkipTests) {
    Invoke-Step 'pytest' { & $python -m pytest }
}

if ($failed.Count -gt 0) {
    Write-Host "`nFAILED: $($failed -join ', ')" -ForegroundColor Red
    exit 1
}

Write-Host "`nAll checks passed." -ForegroundColor Green
exit 0