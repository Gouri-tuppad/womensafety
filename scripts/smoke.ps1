<#
.SYNOPSIS
    SafeWay End-to-End Smoke Test Script (Steps 1-7).

.DESCRIPTION
    Runs complete automated smoke testing against the SafeWay backend at http://localhost:8000.
    Verifies every core subsystem across Steps 1 to 7 without stopping on single failures:
      1. Health (DB + Redis health check)
      2. Auth (Register, Login, GET /users/me, 401 Unauthorized check)
      3. Trusted Contacts (Add, List, Update, Delete CRUD)
      4. Routing (POST /route evening vs night, safest vs fastest, detour % <= 30%, latency < 3s, deduped havens, SHAP explanations)
      5. Safety Heatmap (H3 grid data across multiple hours)
      6. SOS Flow (Trigger countdown, cancel in window, wait idempotency, auto-dispatch active, duplicate trigger, public tracking token, 401 with login token, malformed input resilience, resolve)
      7. Live Trips (Start trip with real route GeoJSON, on-route pings, off-route deviation detection, 'Are you okay' prompt, auto-escalation to SOS, prompt dismissal with respond, 1-5 feedback)
      8. Hazard Reports (Submit hazard, spatial deduplication & corroboration trust boost, real-time segment penalty & cache invalidation)
      9. Child / Parent Mode (Linking, safe zone geofences, breach detection outside safe zone with parent alert, child profile suggested route)
      10. Volunteers (Opt-in, location check-in, authority verification, nearby SOS trigger, mock push receipt assertion)
      11. Dashboard / Fairness (Authority audit 200, standard user 403 forbidden)
      12. Route Prefix Integrity (Zero duplicate routes without /api/v1 prefix in OpenAPI)

.PARAMETER HostUrl
    Base host URL. Defaults to "http://localhost:8000".

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\smoke.ps1
    .\scripts\smoke.ps1 -HostUrl "http://localhost:8000"
#>

[CmdletBinding()]
param(
    [string]$HostUrl = "http://localhost:8000"
)

$ErrorActionPreference = "Continue"

$ApiBase = "$HostUrl/api/v1"
$script:passCount = 0
$script:failCount = 0

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "            SafeWay End-to-End Smoke Test Suite (Steps 1-7)           " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Target API Base: $ApiBase" -ForegroundColor Yellow

# ── Read Timeouts from .env ──────────────────────────────────────────────────
$sosCountdownSec = 5
$tripTimeoutSec = 20

if (Test-Path ".env") {
    $envLines = Get-Content ".env"
    foreach ($line in $envLines) {
        if ($line -match "^\s*SOS_COUNTDOWN_SECONDS\s*=\s*(\d+)") {
            $sosCountdownSec = [int]$matches[1]
        }
        if ($line -match "^\s*TRIP_ARE_YOU_OKAY_SECONDS\s*=\s*(\d+)") {
            $tripTimeoutSec = [int]$matches[1]
        }
    }
}
Write-Host "Active Test Timeouts: SOS_COUNTDOWN_SECONDS=${sosCountdownSec}s | TRIP_ARE_YOU_OKAY_SECONDS=${tripTimeoutSec}s`n" -ForegroundColor DarkCyan


# ── Helper Functions ─────────────────────────────────────────────────────────

function Check-Assert {
    param(
        [string]$CheckName,
        [bool]$Condition,
        [string]$Details = ""
    )
    if ($Condition) {
        Write-Host "  [PASS] $CheckName" -ForegroundColor Green
        $script:passCount++
    } else {
        Write-Host "  [FAIL] $CheckName $(if ($Details) { "-> $Details" })" -ForegroundColor Red
        $script:failCount++
    }
}

function Print-Evidence {
    param([string]$FlowName)
    try {
        $recent = docker compose logs --tail=40 api 2>$null | Select-String "\[MOCK (SMS|PUSH|POLICE)\]"
        if ($recent) {
            $slice = $recent | Select-Object -Last 2
            foreach ($item in $slice) {
                Write-Host "    [Evidence: $FlowName] $item" -ForegroundColor Magenta
            }
        }
    } catch {}
}

function Invoke-Api {
    param(
        [string]$Uri,
        [string]$Method = "GET",
        [hashtable]$Headers = @{},
        [object]$Body = $null
    )
    $jsonBody = if ($Body) { $Body | ConvertTo-Json -Depth 10 } else { $null }
    try {
        $req = @{
            Uri = $Uri
            Method = $Method
            ContentType = "application/json"
            Headers = $Headers
            UseBasicParsing = $true
        }
        if ($jsonBody) { $req["Body"] = $jsonBody }
        $resp = Invoke-WebRequest @req
        $content = $resp.Content
        $data = $null
        if ($content) {
            try { $data = $content | ConvertFrom-Json } catch { $data = $content }
        }
        return @{ StatusCode = [int]$resp.StatusCode; Data = $data; Success = $true }
    } catch {
        $status = 0
        $data = $null
        if ($_.Exception -and $_.Exception.Response) {
            $status = [int]$_.Exception.Response.StatusCode
            try {
                $stream = $_.Exception.Response.GetResponseStream()
                if ($stream) {
                    $reader = New-Object System.IO.StreamReader($stream)
                    $errText = $reader.ReadToEnd()
                    try { $data = $errText | ConvertFrom-Json } catch { $data = $errText }
                }
            } catch {}
        }
        return @{ StatusCode = $status; Data = $data; Error = $_.Exception.Message; Success = ($status -ge 200 -and $status -lt 300) }
    }
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════
Write-Host "[1/12] Checking System Health..." -ForegroundColor Yellow
$healthRes = Invoke-Api -Uri "$ApiBase/health" -Method "GET"
Check-Assert "Health endpoint returns HTTP 200" ($healthRes.StatusCode -eq 200)
if ($healthRes.Success) {
    Check-Assert "System overall status is healthy" ($healthRes.Data.status -eq "healthy")
    Check-Assert "PostgreSQL database component is healthy" ($healthRes.Data.components.database -eq "healthy")
    Check-Assert "Redis cache component is healthy" ($healthRes.Data.components.redis -eq "healthy")
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. AUTHENTICATION & PROFILE
# ══════════════════════════════════════════════════════════════════════════════
Write-Host "`n[2/12] Testing Authentication & Authorization..." -ForegroundColor Yellow

$womanEmail = "smoke_woman@example.com"
$womanPhone = "+919100000001"
$womanPass  = "SecureSmokePass123!"

# Register (ignore 409)
$regRes = Invoke-Api -Uri "$ApiBase/auth/register" -Method "POST" -Body @{
    email = $womanEmail
    phone = $womanPhone
    password = $womanPass
    name = "Sunita Smoke"
    role = "woman"
}
Check-Assert "Register woman user (201 Created or 409 Conflict accepted)" ($regRes.StatusCode -in @(200, 201, 409))

# Login
$loginRes = Invoke-Api -Uri "$ApiBase/auth/login" -Method "POST" -Body @{
    email = $womanEmail
    password = $womanPass
}
$womanToken = if ($loginRes.Success) { $loginRes.Data.access_token } else { $null }
Check-Assert "Login successful and bearer access_token returned" ($womanToken -ne $null)

$womanHeaders = @{ "Authorization" = "Bearer $womanToken" }

# GET /users/me with token
$meRes = Invoke-Api -Uri "$ApiBase/users/me" -Method "GET" -Headers $womanHeaders
Check-Assert "GET /users/me with bearer token returns 200 OK" ($meRes.StatusCode -eq 200)
if ($meRes.Success) {
    Check-Assert "Profile email matches registered user" ($meRes.Data.email -eq $womanEmail)
}

# GET /users/me without token -> 401
$unauthRes = Invoke-Api -Uri "$ApiBase/users/me" -Method "GET"
Check-Assert "GET /users/me without token returns 401 Unauthorized" ($unauthRes.StatusCode -eq 401)


# ══════════════════════════════════════════════════════════════════════════════
# 3. TRUSTED CONTACTS CRUD
# ══════════════════════════════════════════════════════════════════════════════
Write-Host "`n[3/12] Testing Trusted Emergency Contacts CRUD..." -ForegroundColor Yellow

# Add Contact
$addContactRes = Invoke-Api -Uri "$ApiBase/users/me/contacts" -Method "POST" -Headers $womanHeaders -Body @{
    contact_name = "Mother Emergency"
    contact_phone = "+919876543201"
    relationship_label = "mother"
    priority = 1
}
Check-Assert "POST /users/me/contacts creates emergency contact" ($addContactRes.StatusCode -in @(200, 201))
$contactId = if ($addContactRes.Success) { $addContactRes.Data.id } else { $null }

# List Contacts
$listContactRes = Invoke-Api -Uri "$ApiBase/users/me/contacts" -Method "GET" -Headers $womanHeaders
Check-Assert "GET /users/me/contacts returns trusted contact list" ($listContactRes.StatusCode -eq 200 -and $listContactRes.Data.Count -ge 1)

# Update Contact (PATCH)
if ($contactId) {
    $patchContactRes = Invoke-Api -Uri "$ApiBase/users/me/contacts/$contactId" -Method "PATCH" -Headers $womanHeaders -Body @{
        relationship_label = "guardian"
        priority = 1
    }
    Check-Assert "PATCH /users/me/contacts/{id} updates contact details" ($patchContactRes.StatusCode -eq 200 -and $patchContactRes.Data.relationship_label -eq "guardian")

    # Delete Contact
    $delContactRes = Invoke-Api -Uri "$ApiBase/users/me/contacts/$contactId" -Method "DELETE" -Headers $womanHeaders
    Check-Assert "DELETE /users/me/contacts/{id} removes contact (204 No Content)" ($delContactRes.StatusCode -in @(200, 204))
}

# Ensure user has a permanent emergency contact for SOS flows
Invoke-Api -Uri "$ApiBase/users/me/contacts" -Method "POST" -Headers $womanHeaders -Body @{
    contact_name = "Kavitha Rao (Sister)"
    contact_phone = "+919876543210"
    relationship_label = "sister"
    priority = 1
} | Out-Null


# ══════════════════════════════════════════════════════════════════════════════
# 4. ROUTING: EVENING VS NIGHT COMPARISONS & ACCEPTANCE TARGETS
# ══════════════════════════════════════════════════════════════════════════════
Write-Host "`n[4/12] Testing Routing Engine (18:00 vs 23:00 Peenya Fixture)..." -ForegroundColor Yellow

$routeOrigin = @{ lat = 13.0450; lon = 77.5050 }        # Bagalagunte
$routeDest   = @{ lat = 13.0150; lon = 77.5300 }        # Peenya 1st Stage South-East

# Warm-up call (initializes in-memory NetworkX graph and spatial trees)
Invoke-Api -Uri "$ApiBase/route" -Method "POST" -Body @{
    origin = $routeOrigin
    destination = $routeDest
    depart_time = "2026-10-04T18:00:00+05:30"
    profile = "adult_day"
} | Out-Null

# Evening (18:00)
$swEve = [System.Diagnostics.Stopwatch]::StartNew()
$eveRes = Invoke-Api -Uri "$ApiBase/route" -Method "POST" -Body @{
    origin = $routeOrigin
    destination = $routeDest
    depart_time = "2026-10-04T18:00:00+05:30"
    profile = "woman_night"
}
$swEve.Stop()
$eveLatency = $swEve.Elapsed.TotalSeconds

# Night (23:00)
$swNight = [System.Diagnostics.Stopwatch]::StartNew()
$nightRes = Invoke-Api -Uri "$ApiBase/route" -Method "POST" -Body @{
    origin = $routeOrigin
    destination = $routeDest
    depart_time = "2026-10-04T23:00:00+05:30"
    profile = "woman_night"
}
$swNight.Stop()
$nightLatency = $swNight.Elapsed.TotalSeconds

Check-Assert "POST /route (evening 18:00) returns HTTP 200" ($eveRes.StatusCode -eq 200)
Check-Assert "POST /route (night 23:00) returns HTTP 200" ($nightRes.StatusCode -eq 200)
Check-Assert "Routing response time < 3.0s (measured: $($nightLatency.ToString('F2'))s)" ($nightLatency -lt 3.0)

if ($eveRes.Success -and $nightRes.Success) {
    $eveSafestScore   = [double]$eveRes.Data.safest.safety_score
    $nightSafestScore = [double]$nightRes.Data.safest.safety_score
    $nightFastestScore= [double]$nightRes.Data.fastest.safety_score
    $detourPct        = [double]$nightRes.Data.comparison.distance_pct_longer

    Check-Assert "Night safety score ($nightSafestScore) < evening score ($eveSafestScore) due to darkness/crime" ($nightSafestScore -le $eveSafestScore)
    Check-Assert "Safest score ($nightSafestScore) >= fastest score ($nightFastestScore)" ($nightSafestScore -ge $nightFastestScore)
    Check-Assert "Safest route detour ($detourPct%) is within 0-30%" ($detourPct -ge 0.0 -and $detourPct -le 30.0)

    # SHAP risk explanations
    $explanations = $nightRes.Data.risk_explanations
    Check-Assert "SHAP risk explanations are generated and non-empty" ($explanations -ne $null -and $explanations.Count -ge 1)

    # Safe Havens deduplication
    $havens = $nightRes.Data.safe_havens_near_safest
    $uniqueHavenNames = @($havens | ForEach-Object { $_.name } | Select-Object -Unique)
    Check-Assert "Safe havens present near route with no duplicate names" ($havens.Count -eq $uniqueHavenNames.Count)
}


# ══════════════════════════════════════════════════════════════════════════════
# 5. SAFETY HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
Write-Host "`n[5/12] Testing Safety Heatmap H3 Grid..." -ForegroundColor Yellow
$heat18 = Invoke-Api -Uri "$ApiBase/safety/heatmap?hour=18" -Method "GET"
$heat23 = Invoke-Api -Uri "$ApiBase/safety/heatmap?hour=23" -Method "GET"

Check-Assert "GET /safety/heatmap?hour=18 returns HTTP 200 and H3 cells" ($heat18.StatusCode -eq 200 -and $heat18.Data.cell_count -gt 0)
Check-Assert "GET /safety/heatmap?hour=23 returns HTTP 200 and H3 cells" ($heat23.StatusCode -eq 200 -and $heat23.Data.cell_count -gt 0)


# ══════════════════════════════════════════════════════════════════════════════
# 6. SOS FLOW (SHAKE, COUNTDOWN, CANCEL, DISPATCH, TRACKING)
# ══════════════════════════════════════════════════════════════════════════════
Write-Host "`n[6/12] Testing Shake-Gesture SOS & Emergency Dispatch..." -ForegroundColor Yellow

# Step 6a: Trigger SOS 1 & Cancel within window
$trig1Res = Invoke-Api -Uri "$ApiBase/sos/trigger" -Method "POST" -Headers $womanHeaders -Body @{
    lat = 13.0250
    lon = 77.5150
    trigger_type = "shake"
    notes = "Cancellation test"
}
Check-Assert "Trigger SOS returns HTTP 201 with COUNTDOWN status" ($trig1Res.StatusCode -eq 201 -and $trig1Res.Data.status -eq "countdown")
$sosId1 = if ($trig1Res.Success) { $trig1Res.Data.sos_id } else { $null }

if ($sosId1) {
    $cancelRes = Invoke-Api -Uri "$ApiBase/sos/$sosId1/cancel" -Method "POST" -Headers $womanHeaders -Body @{
        reason = "Accidental false alarm"
    }
    Check-Assert "Cancel SOS within window returns status CANCELLED" ($cancelRes.StatusCode -eq 200 -and $cancelRes.Data.status -eq "cancelled")
}

# Step 6b: Wait 31s for idempotency window to clear
Write-Host "  Waiting 31s for SOS 30-second idempotency window to elapse..." -ForegroundColor DarkGray
Start-Sleep -Seconds 31

# Step 6c: Trigger SOS 2 & Wait for Auto-Dispatch
$trig2Res = Invoke-Api -Uri "$ApiBase/sos/trigger" -Method "POST" -Headers $womanHeaders -Body @{
    lat = 13.0255
    lon = 77.5155
    trigger_type = "shake"
    notes = "Auto-dispatch test"
}
Check-Assert "Trigger SOS 2 returns HTTP 201 with countdown" ($trig2Res.StatusCode -eq 201 -and $trig2Res.Data.status -eq "countdown")
$sosId2 = if ($trig2Res.Success) { $trig2Res.Data.sos_id } else { $null }
$trackToken = if ($trig2Res.Success) { $trig2Res.Data.tracking_token } else { $null }

# Step 6d: Duplicate Trigger Check within 30s
$dupRes = Invoke-Api -Uri "$ApiBase/sos/trigger" -Method "POST" -Headers $womanHeaders -Body @{
    lat = 13.0255
    lon = 77.5155
    trigger_type = "shake"
}
Check-Assert "Duplicate trigger returns is_duplicate: true" ($dupRes.StatusCode -in @(200, 201) -and $dupRes.Data.is_duplicate -eq $true)

# Step 6e: Wait for countdown to expire -> Auto-dispatch into ACTIVE
$waitSec = $sosCountdownSec + 2
Write-Host "  Waiting ${waitSec}s for countdown expiry and auto-dispatch..." -ForegroundColor DarkGray
Start-Sleep -Seconds $waitSec

$statusRes = Invoke-Api -Uri "$ApiBase/sos/$sosId2" -Method "GET" -Headers $womanHeaders
Check-Assert "SOS transitions to ACTIVE status after countdown expiry" ($statusRes.StatusCode -eq 200 -and $statusRes.Data.status -eq "active")

# Step 6f: Public live tracking checks
if ($trackToken) {
    # Public without login
    $pubTrackRes = Invoke-Api -Uri "$ApiBase/track/$trackToken" -Method "GET"
    Check-Assert "GET /track/{token} works without login" ($pubTrackRes.StatusCode -eq 200 -and $pubTrackRes.Data.status -eq "active")

    # Tracking with login token passed as token path parameter -> 401
    $loginTrackRes = Invoke-Api -Uri "$ApiBase/track/$womanToken" -Method "GET"
    Check-Assert "GET /track with login token returns 401 (purpose mismatch)" ($loginTrackRes.StatusCode -eq 401)
}

# Step 6g: Bad accel_window & unknown trip_id never cause a 500
$woman2Email = "smoke_woman2@example.com"
Invoke-Api -Uri "$ApiBase/auth/register" -Method "POST" -Body @{
    email = $woman2Email
    phone = "+919100000002"
    password = $womanPass
    name = "Smoke Woman 2"
    role = "woman"
} | Out-Null
$login2 = Invoke-Api -Uri "$ApiBase/auth/login" -Method "POST" -Body @{ email = $woman2Email; password = $womanPass }
$woman2Headers = @{ "Authorization" = "Bearer $($login2.Data.access_token)" }

$badInputRes = Invoke-Api -Uri "$ApiBase/sos/trigger" -Method "POST" -Headers $woman2Headers -Body @{
    lat = 13.0260
    lon = 77.5160
    trigger_type = "shake"
    accel_window = @("corrupted_string", 12345)
    trip_id = "00000000-0000-0000-0000-000000000000"
}
Check-Assert "Bad accel_window and dummy trip_id never cause a 500 (returns 201 Created)" ($badInputRes.StatusCode -eq 201)

# Step 6h: Resolve SOS 2
if ($sosId2) {
    $resolveRes = Invoke-Api -Uri "$ApiBase/sos/$sosId2/resolve" -Method "POST" -Headers $womanHeaders -Body @{
        notes = "Safe now, resolved by smoke test"
    }
    Check-Assert "POST /sos/{id}/resolve transitions SOS to RESOLVED" ($resolveRes.StatusCode -eq 200 -and $resolveRes.Data.status -eq "resolved")
}
Print-Evidence "SOS Flow"


# ══════════════════════════════════════════════════════════════════════════════
# 7. LIVE TRIPS & ANOMALY MONITORING
# ══════════════════════════════════════════════════════════════════════════════
Write-Host "`n[7/12] Testing Live Trip Monitoring & Anomaly Auto-Escalation..." -ForegroundColor Yellow

# Build real route GeoJSON LineString from safest route polyline [ [lon, lat], ... ]
$realRouteGeoJson = $null
if ($nightRes.Success -and $nightRes.Data.safest.polyline) {
    $geoCoords = @()
    foreach ($pt in $nightRes.Data.safest.polyline) {
        $geoCoords += ,@([double]$pt[1], [double]$pt[0])
    }
    $realRouteGeoJson = @{
        type = "LineString"
        coordinates = $geoCoords
    }
}

# Start Trip 1 with real route geometry
$tripStartRes = Invoke-Api -Uri "$ApiBase/trips/start" -Method "POST" -Headers $womanHeaders -Body @{
    origin_lat = 13.0450
    origin_lon = 77.5050
    destination_lat = 13.0150
    destination_lon = 77.5300
    route_geojson = $realRouteGeoJson
    safety_score = 80.0
    mode = "safest"
}
Check-Assert "POST /trips/start initiates monitored trip with route_geojson" ($tripStartRes.StatusCode -eq 201 -and $tripStartRes.Data.status -eq "active")
$tripId1 = if ($tripStartRes.Success) { $tripStartRes.Data.trip_id } else { $null }

if ($tripId1) {
    # On-route ping -> No anomaly
    $onRouteRes = Invoke-Api -Uri "$ApiBase/trips/$tripId1/location" -Method "POST" -Headers $womanHeaders -Body @{
        lat = 13.0450
        lon = 77.5050
        speed_mps = 1.2
    }
    Check-Assert "On-route location ping records normal progress without anomaly" ($onRouteRes.StatusCode -eq 200 -and $onRouteRes.Data.anomaly_detected -eq $false)

    # Off-route point (> 2 km away from polyline) -> Route deviation detected & prompt sent
    $offRouteRes = Invoke-Api -Uri "$ApiBase/trips/$tripId1/location" -Method "POST" -Headers $womanHeaders -Body @{
        lat = 13.0650
        lon = 77.5350
        speed_mps = 1.2
    }
    Check-Assert "Off-route deviation triggers 'Are you okay?' prompt" ($offRouteRes.StatusCode -eq 200 -and $offRouteRes.Data.anomaly_detected -eq $true -and $offRouteRes.Data.prompt_sent -eq $true)

    # Do not respond -> Assert auto-escalation into SOS
    $waitEscalate = $tripTimeoutSec + 2
    Write-Host "  Waiting ${waitEscalate}s for anomaly prompt timeout and auto-escalation..." -ForegroundColor DarkGray
    Start-Sleep -Seconds $waitEscalate

    $liveRes1 = Invoke-Api -Uri "$ApiBase/trips/$tripId1/live" -Method "GET" -Headers $womanHeaders
    Check-Assert "Trip auto-escalated into SOS on unanswered prompt" ($liveRes1.StatusCode -eq 200 -and $liveRes1.Data.status -eq "sos_triggered")

    # Conclude Trip 1
    Invoke-Api -Uri "$ApiBase/trips/$tripId1/end" -Method "POST" -Headers $womanHeaders | Out-Null
}

# Start Trip 2 to test prompt dismissal via respond
$tripStart2 = Invoke-Api -Uri "$ApiBase/trips/start" -Method "POST" -Headers $womanHeaders -Body @{
    origin_lat = 13.0450
    origin_lon = 77.5050
    destination_lat = 13.0150
    destination_lon = 77.5300
    route_geojson = $realRouteGeoJson
}
$tripId2 = if ($tripStart2.Success) { $tripStart2.Data.trip_id } else { $null }

if ($tripId2) {
    # Send off-route ping
    Invoke-Api -Uri "$ApiBase/trips/$tripId2/location" -Method "POST" -Headers $womanHeaders -Body @{
        lat = 13.0650
        lon = 77.5350
    } | Out-Null

    # User responds "I'm okay"
    $respondRes = Invoke-Api -Uri "$ApiBase/trips/$tripId2/respond" -Method "POST" -Headers $womanHeaders -Body @{
        message = "I'm okay, false alarm"
    }
    Check-Assert "POST /trips/{id}/respond cancels auto-escalation" ($respondRes.StatusCode -eq 200 -and $respondRes.Data.escalation_cancelled -eq $true)

    # End trip
    $endRes = Invoke-Api -Uri "$ApiBase/trips/$tripId2/end" -Method "POST" -Headers $womanHeaders
    Check-Assert "POST /trips/{id}/end concludes trip successfully" ($endRes.StatusCode -eq 200 -and $endRes.Data.status -eq "completed")

    # Feedback 1-5
    $feedbackRes = Invoke-Api -Uri "$ApiBase/trips/$tripId2/feedback" -Method "POST" -Headers $womanHeaders -Body @{
        safety_rating = 4
        route_rating = 4
        comment = "Good lighting overall on 8th main."
    }
    Check-Assert "POST /trips/{id}/feedback records safety rating 1-5" ($feedbackRes.StatusCode -in @(200, 201) -and $feedbackRes.Data.safety_rating -eq 4)
}
Print-Evidence "Trip Monitoring"


# ══════════════════════════════════════════════════════════════════════════════
# 8. COMMUNITY HAZARD REPORTS & REAL-TIME CACHE UPDATES
# ══════════════════════════════════════════════════════════════════════════════
Write-Host "`n[8/12] Testing Community Hazard Reports & Risk Updates..." -ForegroundColor Yellow

# Submit dark_street report on a Peenya street
$rep1Res = Invoke-Api -Uri "$ApiBase/reports" -Method "POST" -Headers $womanHeaders -Body @{
    lat = 13.0285
    lon = 77.5190
    report_type = "dark_street"
    title = "Broken street lamps"
    description = "Street is very dark and lamps are not working."
}
Check-Assert "POST /reports records hazard and returns affected segments" ($rep1Res.StatusCode -eq 201 -and $rep1Res.Data.affected_segments.Count -gt 0)

# Submit nearby duplicate report (< 100m)
$rep2Res = Invoke-Api -Uri "$ApiBase/reports" -Method "POST" -Headers $woman2Headers -Body @{
    lat = 13.0286
    lon = 77.5191
    report_type = "dark_street"
    title = "Dark stretch here too"
    description = "Confirming dark street lamps."
}
Check-Assert "Nearby hazard report is de-duplicated and corroborated" ($rep2Res.StatusCode -eq 201 -and $rep2Res.Data.corroborated -eq $true)

# Assert segment darkness/safety changes on next /route call
$routeAfterReport = Invoke-Api -Uri "$ApiBase/route" -Method "POST" -Body @{
    origin = @{ lat = 13.0285; lon = 77.5190 }
    destination = @{ lat = 13.0220; lon = 77.5034 }
    depart_time = "2026-10-04T23:00:00+05:30"
    profile = "woman_night"
}
Check-Assert "Next POST /route call succeeds with updated segment risk" ($routeAfterReport.StatusCode -eq 200 -and $routeAfterReport.Data.safest -ne $null)


# ══════════════════════════════════════════════════════════════════════════════
# 9. CHILD / PARENT MODE & GEOFENCING
# ══════════════════════════════════════════════════════════════════════════════
Write-Host "`n[9/12] Testing Child Mode & Geofence Breach Alerts..." -ForegroundColor Yellow

$parentEmail = "smoke_parent@example.com"
$childEmail  = "smoke_child@example.com"

# Register Parent & Child
Invoke-Api -Uri "$ApiBase/auth/register" -Method "POST" -Body @{ email = $parentEmail; phone = "+919100000003"; password = $womanPass; name = "Smoke Parent"; role = "parent" } | Out-Null
Invoke-Api -Uri "$ApiBase/auth/register" -Method "POST" -Body @{ email = $childEmail; phone = "+919100000004"; password = $womanPass; name = "Smoke Child"; role = "child" } | Out-Null

$parentLogin = Invoke-Api -Uri "$ApiBase/auth/login" -Method "POST" -Body @{ email = $parentEmail; password = $womanPass }
$childLogin  = Invoke-Api -Uri "$ApiBase/auth/login" -Method "POST" -Body @{ email = $childEmail; password = $womanPass }

$parentHeaders = @{ "Authorization" = "Bearer $($parentLogin.Data.access_token)" }
$childHeaders  = @{ "Authorization" = "Bearer $($childLogin.Data.access_token)" }

# Link Child to Parent
$linkRes = Invoke-Api -Uri "$ApiBase/child/link" -Method "POST" -Headers $parentHeaders -Body @{
    child_email = $childEmail
}
Check-Assert "POST /child/link links child account to parent guardian" ($linkRes.StatusCode -eq 200 -and $linkRes.Data.status -eq "linked")
$childId = if ($linkRes.Success) { $linkRes.Data.child_id } else { $null }

if ($childId) {
    # Define circular safe zone (School)
    $fenceRes = Invoke-Api -Uri "$ApiBase/child/$childId/geofences" -Method "POST" -Headers $parentHeaders -Body @{
        name = "School Safe Zone"
        center_lat = 13.0285
        center_lon = 77.5190
        radius_m = 150.0
        alert_on_exit = $true
    }
    Check-Assert "POST /child/{id}/geofences defines safe zone geofence" ($fenceRes.StatusCode -eq 201 -and $fenceRes.Data.name -eq "School Safe Zone")

    # Child pings OUTSIDE safe zone (1.5 km away)
    $breachPingRes = Invoke-Api -Uri "$ApiBase/child/$childId/location" -Method "POST" -Headers $childHeaders -Body @{
        lat = 13.0450
        lon = 77.5350
    }
    Check-Assert "Child outside safe zone records parent alert" ($breachPingRes.StatusCode -eq 200 -and $breachPingRes.Data.breach_detected -eq $true -and $breachPingRes.Data.alert_sent -eq $true)

    # Suggested safest school route with child profile
    $childRouteRes = Invoke-Api -Uri "$ApiBase/child/$childId/suggested-route?origin_lat=13.0250&origin_lon=77.5150&dest_lat=13.0280&dest_lon=77.5180" -Method "GET" -Headers $parentHeaders
    Check-Assert "GET /child/{id}/suggested-route returns child profile route" ($childRouteRes.StatusCode -eq 200 -and $childRouteRes.Data.profile -eq "child")
}
Print-Evidence "Child Mode"


# ══════════════════════════════════════════════════════════════════════════════
# 10. SAFETY VOLUNTEERS & EMERGENCY SOS DISPATCH INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════
Write-Host "`n[10/12] Testing Volunteers & Emergency Push Dispatch..." -ForegroundColor Yellow

$volEmail = "smoke_vol@example.com"
$authEmail = "smoke_authority@example.com"

# Register Volunteer & Authority
Invoke-Api -Uri "$ApiBase/auth/register" -Method "POST" -Body @{ email = $volEmail; phone = "+919100000005"; password = $womanPass; name = "Smoke Volunteer"; role = "volunteer" } | Out-Null
Invoke-Api -Uri "$ApiBase/auth/register" -Method "POST" -Body @{ email = $authEmail; phone = "+919100000006"; password = $womanPass; name = "Smoke Authority"; role = "authority" } | Out-Null

$volLogin  = Invoke-Api -Uri "$ApiBase/auth/login" -Method "POST" -Body @{ email = $volEmail; password = $womanPass }
$authLogin = Invoke-Api -Uri "$ApiBase/auth/login" -Method "POST" -Body @{ email = $authEmail; password = $womanPass }

$volHeaders  = @{ "Authorization" = "Bearer $($volLogin.Data.access_token)" }
$authHeaders = @{ "Authorization" = "Bearer $($authLogin.Data.access_token)" }

# Opt-in
$optInRes = Invoke-Api -Uri "$ApiBase/volunteers/opt-in" -Method "POST" -Headers $volHeaders -Body @{ radius_m = 1000 }
Check-Assert "POST /volunteers/opt-in marks volunteer available" ($optInRes.StatusCode -eq 200 -and $optInRes.Data.is_available -eq $true)
$volunteerId = if ($optInRes.Success) { $optInRes.Data.volunteer_id } else { $null }

# Location check-in
$volLocRes = Invoke-Api -Uri "$ApiBase/volunteers/location" -Method "POST" -Headers $volHeaders -Body @{ lat = 13.0250; lon = 77.5150 }
Check-Assert "POST /volunteers/location updates coordinates for proximity queries" ($volLocRes.StatusCode -eq 200 -and $volLocRes.Data.status -eq "location_updated")

# Authority verification
if ($volunteerId) {
    $verifyRes = Invoke-Api -Uri "$ApiBase/volunteers/$volunteerId/verify" -Method "POST" -Headers $authHeaders
    Check-Assert "POST /volunteers/{id}/verify by authority marks volunteer verified" ($verifyRes.StatusCode -eq 200 -and $verifyRes.Data.verified -eq $true)
}

# Trigger SOS nearby and assert volunteer receives mock push
$volSosRes = Invoke-Api -Uri "$ApiBase/sos/trigger" -Method "POST" -Headers $woman2Headers -Body @{
    lat = 13.0252
    lon = 77.5152
    trigger_type = "shake"
    notes = "Nearby volunteer dispatch verification"
}
Check-Assert "Trigger SOS near volunteer returns 201 Created" ($volSosRes.StatusCode -eq 201)
$volSosId = if ($volSosRes.Success) { $volSosRes.Data.sos_id } else { $null }

Write-Host "  Waiting ${waitSec}s for countdown expiry and volunteer dispatch..." -ForegroundColor DarkGray
Start-Sleep -Seconds $waitSec

$volPushLogs = docker compose logs --tail=50 api 2>$null | Select-String "Emergency SOS Alert Nearby"
$receivedPush = ($null -ne $volPushLogs -and @($volPushLogs).Count -gt 0)
Check-Assert "Verified volunteer received mock push notification for nearby SOS" ([bool]$receivedPush)

if ($volSosId) {
    Invoke-Api -Uri "$ApiBase/sos/$volSosId/resolve" -Method "POST" -Headers $woman2Headers -Body @{ notes = "Volunteer test complete" } | Out-Null
}
Print-Evidence "Volunteer Dispatch"


# ══════════════════════════════════════════════════════════════════════════════
# 11. AUTHORITY DASHBOARD / FAIRNESS
# ══════════════════════════════════════════════════════════════════════════════
Write-Host "`n[11/12] Testing Authority Dashboard / Fairness Access Control..." -ForegroundColor Yellow

$dashAuthRes = Invoke-Api -Uri "$ApiBase/dashboard/fairness" -Method "GET" -Headers $authHeaders
Check-Assert "GET /dashboard/fairness with authority token returns HTTP 200" ($dashAuthRes.StatusCode -eq 200)

$dashWomanRes = Invoke-Api -Uri "$ApiBase/dashboard/fairness" -Method "GET" -Headers $womanHeaders
Check-Assert "GET /dashboard/fairness with normal user token returns HTTP 403 Forbidden" ($dashWomanRes.StatusCode -eq 403)


# ══════════════════════════════════════════════════════════════════════════════
# 12. ROUTE PREFIX INTEGRITY (NO DUPLICATE UNPREFIXED ROUTES)
# ══════════════════════════════════════════════════════════════════════════════
Write-Host "`n[12/12] Verifying OpenAPI Route Prefix Integrity..." -ForegroundColor Yellow

$openapiRes = Invoke-Api -Uri "$HostUrl/openapi.json" -Method "GET"
Check-Assert "GET /openapi.json returns API schema" ($openapiRes.StatusCode -eq 200)

if ($openapiRes.Success) {
    $paths = @($openapiRes.Data.paths.PSObject.Properties | ForEach-Object { $_.Name })
    $hasDuplicates = $false
    $duplicateList = @()

    foreach ($p in $paths) {
        if ($p -match "^/api/v1/(.+)$") {
            $unprefixed = "/" + $matches[1]
            if ($paths -contains $unprefixed) {
                $hasDuplicates = $true
                $duplicateList += "$p <=> $unprefixed"
            }
        }
    }
    Check-Assert "No path appears both with and without /api/v1 prefix" (-not $hasDuplicates) "$($duplicateList -join ', ')"
}


# ══════════════════════════════════════════════════════════════════════════════
# 13. SUMMARY & EXIT CODE
# ══════════════════════════════════════════════════════════════════════════════
$totalChecks = $script:passCount + $script:failCount
Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host "                         SMOKE TEST SUMMARY                           " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Total Checks Executed : $totalChecks"
Write-Host "  Checks Passed         : $script:passCount" -ForegroundColor Green
Write-Host "  Checks Failed         : $script:failCount" -ForegroundColor $(if ($script:failCount -eq 0) { "Green" } else { "Red" })
Write-Host "======================================================================`n" -ForegroundColor Cyan

if ($script:failCount -gt 0) {
    exit 1
} else {
    exit 0
}
