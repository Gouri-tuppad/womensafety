<#
.SYNOPSIS
    Runs all SafeWay backend verification suites in sequence and prints a unified summary table.

.DESCRIPTION
    Executes in order:
      1. docker compose ps (container availability & health)
      2. pytest -q (full test suite execution inside the api container)
      3. scripts/smoke.ps1 (end-to-end API smoke tests against localhost:8000)

    At conclusion, prints a consolidated summary table detailing container health,
    pytest pass/fail metrics, and smoke test pass/fail metrics.

.PARAMETER HostUrl
    Base URL for the API server. Defaults to "http://localhost:8000".

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run_all_checks.ps1
    .\scripts\run_all_checks.ps1 -HostUrl "http://localhost:8000"
#>

[CmdletBinding()]
param(
    [string]$HostUrl = "http://localhost:8000"
)

$ErrorActionPreference = "Continue"

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "           SafeWay Backend Unified Verification (Steps 1-7)           " -ForegroundColor Cyan
Write-Host "======================================================================`n" -ForegroundColor Cyan

# ── 1. DOCKER COMPOSE PS ──────────────────────────────────────────────────────
Write-Host ">>> [1/3] Checking Docker Compose Services..." -ForegroundColor Yellow
$composeOutput = docker compose ps 2>&1 | Out-String
Write-Host $composeOutput

$apiHealthy   = $composeOutput -match "api"   -and ($composeOutput -match "Up|running")
$dbHealthy    = $composeOutput -match "db"    -and ($composeOutput -match "Up|running")
$redisHealthy = $composeOutput -match "redis" -and ($composeOutput -match "Up|running")

$containersOk = $apiHealthy -and $dbHealthy -and $redisHealthy


# ── 2. PYTEST SUITE ──────────────────────────────────────────────────────────
Write-Host "`n>>> [2/3] Running Pytest Suite inside api container (pytest -q)..." -ForegroundColor Yellow

$pytestPassed = 0
$pytestFailed = 0
$pytestSummary = "Failed to run pytest"
$pytestSuccess = $false

$pytestRaw = docker compose exec -T api pytest -q 2>&1
$pytestOutput = $pytestRaw | Out-String
Write-Host $pytestOutput

foreach ($line in ($pytestOutput -split "`r?`n")) {
    if ($line -match "(\d+)\s+passed") {
        $pytestPassed = [int]$matches[1]
    }
    if ($line -match "(\d+)\s+failed") {
        $pytestFailed = [int]$matches[1]
    }
    if ($line -match "(\d+)\s+passed.*in\s+([\d\.]+)s" -or $line -match "passed.*in") {
        $pytestSummary = $line.Trim()
    }
}
if ($pytestFailed -eq 0 -and $pytestPassed -gt 0) {
    $pytestSuccess = $true
}


# ── 3. SMOKE TESTS (smoke.ps1) ────────────────────────────────────────────────
Write-Host "`n>>> [3/3] Running Smoke Test Suite (scripts/smoke.ps1)..." -ForegroundColor Yellow

$smokePassed = 0
$smokeFailed = 0
$smokeExitCode = 1

$smokeRaw = & powershell.exe -ExecutionPolicy Bypass -File "$PSScriptRoot\smoke.ps1" -HostUrl "$HostUrl" 2>&1
$smokeExitCode = $LASTEXITCODE

foreach ($line in $smokeRaw) {
    Write-Host $line
    if ($line -match "Checks Passed\s*:\s*(\d+)") {
        $smokePassed = [int]$matches[1]
    }
    if ($line -match "Checks Failed\s*:\s*(\d+)") {
        $smokeFailed = [int]$matches[1]
    }
}

$smokeSuccess = ($smokeExitCode -eq 0 -and $smokeFailed -eq 0)


# ── 4. CONSOLIDATED SUMMARY TABLE ─────────────────────────────────────────────
Write-Host "`n==========================================================================================" -ForegroundColor Cyan
Write-Host "                               CONSOLIDATED VERIFICATION TABLE                            " -ForegroundColor Cyan
Write-Host "==========================================================================================" -ForegroundColor Cyan
Write-Host ("{0,-28} | {1,-38} | {2,-16}" -f "Component / Test Suite", "Details / Counts", "Status")
Write-Host ("-" * 88)

# Containers
$containerDetails = if ($containersOk) { "api, db, redis running & healthy" } else { "One or more containers down" }
$containerStatus  = if ($containersOk) { "[PASS]" } else { "[FAIL]" }
Write-Host ("{0,-28} | {1,-38} | " -f "Docker Compose Services", $containerDetails) -NoNewline
Write-Host ("{0,-16}" -f $containerStatus) -ForegroundColor $(if ($containersOk) { "Green" } else { "Red" })

# Pytest
$pytestDetails = if ($pytestPassed -gt 0) { "$pytestPassed passed, $pytestFailed failed" } else { $pytestSummary }
$pytestStatus  = if ($pytestSuccess) { "[PASS]" } else { "[FAIL]" }
Write-Host ("{0,-28} | {1,-38} | " -f "Pytest Suite (Docker)", $pytestDetails) -NoNewline
Write-Host ("{0,-16}" -f $pytestStatus) -ForegroundColor $(if ($pytestSuccess) { "Green" } else { "Red" })

# Smoke tests
$smokeDetails = "$smokePassed passed, $smokeFailed failed"
$smokeStatus  = if ($smokeSuccess) { "[PASS]" } else { "[FAIL]" }
Write-Host ("{0,-28} | {1,-38} | " -f "E2E Smoke Tests (smoke.ps1)", $smokeDetails) -NoNewline
Write-Host ("{0,-16}" -f $smokeStatus) -ForegroundColor $(if ($smokeSuccess) { "Green" } else { "Red" })

Write-Host ("=" * 88) -ForegroundColor Cyan

$allOk = $containersOk -and $pytestSuccess -and $smokeSuccess

if ($allOk) {
    Write-Host "`n>>> FINAL VERDICT: ALL CHECKS PASSED (System fully verified end-to-end)`n" -ForegroundColor Green
    exit 0
} else {
    Write-Host "`n>>> FINAL VERDICT: VERIFICATION FAILURES DETECTED`n" -ForegroundColor Red
    exit 1
}
