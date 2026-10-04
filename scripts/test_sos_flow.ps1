# scripts/test_sos_flow.ps1 - End-to-end testing script for SafeWay SOS flow
param(
    [string]$BaseUrl = "http://localhost:8000"
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "      SafeWay End-to-End SOS Flow Verification Script     " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Target API: $BaseUrl" -ForegroundColor Yellow

$testEmail = "sos_flow_verifier@example.com"
$testPhone = "+919988776655"
$testPassword = "SecurePassword123!"

# ── 1. Register User (ignore 409 if already registered) ───────
Write-Host "`n[Step 1] Registering user ($testEmail)..." -ForegroundColor Yellow
$regBody = @{
    email = $testEmail
    phone = $testPhone
    password = $testPassword
    name = "Priya Sharma"
    role = "woman"
} | ConvertTo-Json

try {
    $regResp = Invoke-RestMethod -Uri "$BaseUrl/api/v1/auth/register" -Method Post -Body $regBody -ContentType "application/json"
    Write-Host "User registered successfully: ID $($regResp.id)" -ForegroundColor Green
} catch {
    $statusCode = $_.Exception.Response.StatusCode.value__
    if ($statusCode -eq 409) {
        Write-Host "User already registered (HTTP 409) - continuing..." -ForegroundColor Gray
    } else {
        Write-Error "Registration failed: $_"
    }
}

# ── 2. Log in ─────────────────────────────────────────────────
Write-Host "`n[Step 2] Logging in via /api/v1/auth/login..." -ForegroundColor Yellow
$loginBody = @{
    email = $testEmail
    password = $testPassword
} | ConvertTo-Json

$loginResp = Invoke-RestMethod -Uri "$BaseUrl/api/v1/auth/login" -Method Post -Body $loginBody -ContentType "application/json"
$token = $loginResp.access_token
if (-not $token) {
    Write-Error "Failed to obtain access token."
}
Write-Host "Login successful! Bearer token acquired." -ForegroundColor Green

$authHeaders = @{
    "Authorization" = "Bearer $token"
}

# ── 3. Add Trusted Contact ────────────────────────────────────
Write-Host "`n[Step 3] Adding trusted emergency contact..." -ForegroundColor Yellow
$contactBody = @{
    contact_name = "Ananya Sharma (Sister)"
    contact_phone = "+919876543210"
    relationship_label = "sister"
    priority = 1
} | ConvertTo-Json

try {
    $contactResp = Invoke-RestMethod -Uri "$BaseUrl/api/v1/users/me/contacts" -Method Post -Headers $authHeaders -Body $contactBody -ContentType "application/json"
    Write-Host "Trusted contact added: $($contactResp.contact_name) ($($contactResp.contact_phone))" -ForegroundColor Green
} catch {
    Write-Host "Note: Contact may already exist, proceeding..." -ForegroundColor Gray
}

# ── 4. Trigger SOS & Cancel within Window ─────────────────────
Write-Host "`n[Step 4] Triggering SOS 1 (to test cancellation in countdown window)..." -ForegroundColor Yellow
$trig1Body = @{
    lat = 13.0250
    lon = 77.5150
    trigger_type = "shake"
    notes = "Cancellation test"
} | ConvertTo-Json

$trig1Resp = Invoke-RestMethod -Uri "$BaseUrl/api/v1/sos/trigger" -Method Post -Headers $authHeaders -Body $trig1Body -ContentType "application/json"
$sosId1 = $trig1Resp.sos_id
Write-Host "SOS 1 created: ID=$sosId1 | status=$($trig1Resp.status) | countdown=$($trig1Resp.countdown_seconds)s" -ForegroundColor Green

Write-Host "Cancelling SOS 1 within window..." -ForegroundColor Yellow
$cancelBody = @{
    reason = "Accidental shake gesture"
} | ConvertTo-Json

$cancelResp = Invoke-RestMethod -Uri "$BaseUrl/api/v1/sos/$sosId1/cancel" -Method Post -Headers $authHeaders -Body $cancelBody -ContentType "application/json"
Write-Host "Cancellation result: status=$($cancelResp.status) | message=$($cancelResp.message)" -ForegroundColor Green

# ── 5. Wait 31s for Idempotency Window ────────────────────────
Write-Host "`n[Step 5] Waiting 31 seconds to clear the 30-second idempotency window..." -ForegroundColor Yellow
for ($i = 31; $i -gt 0; $i--) {
    Write-Host -NoNewline "`rWaiting... ${i}s remaining "
    Start-Sleep -Seconds 1
}
Write-Host "`nIdempotency window elapsed." -ForegroundColor Green

# ── 6. Trigger SOS 2 & Wait for Auto-Dispatch ──────────────────
Write-Host "`n[Step 6] Triggering SOS 2 (to test auto-dispatch after countdown)..." -ForegroundColor Yellow
$trig2Body = @{
    lat = 13.0255
    lon = 77.5155
    trigger_type = "shake"
    notes = "Auto-dispatch test in Peenya Industrial Area"
} | ConvertTo-Json

$trig2Resp = Invoke-RestMethod -Uri "$BaseUrl/api/v1/sos/trigger" -Method Post -Headers $authHeaders -Body $trig2Body -ContentType "application/json"
$sosId2 = $trig2Resp.sos_id
$countdownSec = $trig2Resp.countdown_seconds
$trackToken = $trig2Resp.tracking_token
$trackUrl = $trig2Resp.tracking_url

Write-Host "SOS 2 created: ID=$sosId2 | status=$($trig2Resp.status) | countdown=${countdownSec}s" -ForegroundColor Green
Write-Host "Tracking URL: $trackUrl" -ForegroundColor Cyan

$waitSec = $countdownSec + 2
Write-Host "Waiting $waitSec seconds for countdown to expire and auto-dispatch to run..." -ForegroundColor Yellow
Start-Sleep -Seconds $waitSec

# ── 7. Check Status via GET /api/v1/sos/{id} (Owner check) ─────
Write-Host "`n[Step 7] Checking status via owner endpoint GET /api/v1/sos/$sosId2..." -ForegroundColor Yellow
$statusResp = Invoke-RestMethod -Uri "$BaseUrl/api/v1/sos/$sosId2" -Method Get -Headers $authHeaders
Write-Host "Status Response:" -ForegroundColor Green
$statusResp | Format-List | Out-String | Write-Host

# ── 8. Call Public Live Tracking Endpoint ─────────────────────
Write-Host "`n[Step 8] Calling public tracking endpoint: GET $BaseUrl$trackUrl..." -ForegroundColor Yellow
# Notice: no Authorization header here!
$publicTrackResp = Invoke-RestMethod -Uri "$BaseUrl$trackUrl" -Method Get
Write-Host "Public Tracking Response:" -ForegroundColor Green
$publicTrackResp | Format-List | Out-String | Write-Host

# ── 9. Print Mock SMS Lines from Logs ─────────────────────────
Write-Host "`n[Step 9] Inspecting last Mock SMS dispatches from Docker logs..." -ForegroundColor Yellow
$smsLogs = docker compose logs --tail=100 api | Select-String "MOCK SMS"
if ($smsLogs) {
    $smsLogs | Select-Object -Last 10 | ForEach-Object { Write-Host $_ -ForegroundColor Magenta }
} else {
    Write-Host "No MOCK SMS lines found in recent logs." -ForegroundColor Gray
}

Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host "              SOS Flow Verification Completed!            " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
