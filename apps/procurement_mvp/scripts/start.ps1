$ErrorActionPreference = "Stop"

$MvpRoot = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $MvpRoot "backend"
$Frontend = Join-Path $MvpRoot "frontend"
$VenvPython = Join-Path $MvpRoot ".venv\Scripts\python.exe"
$NpmCache = Join-Path $MvpRoot ".cache\npm"
$LogDir = Join-Path $MvpRoot ".cache\logs"
$BackendLog = Join-Path $LogDir "backend.log"
$FrontendLog = Join-Path $LogDir "frontend.log"
$PortableNodeRoot = "E:\DevTools\nodejs"

New-Item -ItemType Directory -Force -Path $NpmCache, $LogDir | Out-Null

$NodeDir = $null
$npmCmd = Get-Command npm.cmd -ErrorAction SilentlyContinue
if ($npmCmd) {
    $NodeDir = Split-Path -Parent $npmCmd.Source
} elseif (Test-Path $PortableNodeRoot) {
    $PortableNode = Get-ChildItem $PortableNodeRoot -Directory -Filter "node-*-win-x64" |
        Where-Object { Test-Path (Join-Path $_.FullName "npm.cmd") } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($PortableNode) {
        $NodeDir = $PortableNode.FullName
    }
}

if (-not $NodeDir -or -not (Test-Path (Join-Path $NodeDir "npm.cmd"))) {
    throw "Node.js/npm was not found. Install Node.js 20+ under E:\DevTools\nodejs or add npm to PATH."
}
if (-not (Test-Path $VenvPython)) {
    throw "Missing D-drive venv. Run .\scripts\setup.ps1 first."
}
if (-not (Test-Path (Join-Path $Frontend "node_modules"))) {
    throw "Missing frontend dependencies. Run .\scripts\setup.ps1 first."
}

$env:Path = "$NodeDir;$env:Path"
$env:npm_config_cache = $NpmCache
$Npm = Join-Path $NodeDir "npm.cmd"

Write-Host "Initializing demo database..."
& $VenvPython (Join-Path $PSScriptRoot "init_demo.py")

function Test-Url([string]$Url) {
    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 500
    } catch {
        return $false
    }
}

function Test-BackendFresh {
    try {
        $openapi = (Invoke-WebRequest -Uri "http://127.0.0.1:8000/openapi.json" -UseBasicParsing -TimeoutSec 3).Content
        return $openapi -match "submit-procurement" -and $openapi -match "erp/suppliers"
    } catch {
        return $false
    }
}

function Stop-PortListener([int]$Port) {
    $pids = @()
    try {
        $pids = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique
    } catch {}
    foreach ($procId in $pids) {
        if (-not $procId) { continue }
        Write-Host "Stopping stale process on port $Port (PID $procId)"
        cmd /c "taskkill /PID $procId /T /F" | Out-Null
        Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -and $_.CommandLine -like "*parent_pid=$procId*" } |
            ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    }
}

$backendFresh = (Test-Url "http://127.0.0.1:8000/health") -and (Test-BackendFresh)
if ($backendFresh) {
    Write-Host "Backend already running with procurement-cloud APIs at http://127.0.0.1:8000"
} else {
    if (Test-Url "http://127.0.0.1:8000/health") {
        Write-Host "Backend on :8000 is stale (missing submit-procurement / erp/suppliers). Restarting..."
        Stop-PortListener 8000
        Start-Sleep -Seconds 2
    }
    Write-Host "Starting backend -> $BackendLog"
    $backendCmd = @"
`$ErrorActionPreference='Continue'
Set-Location '$Backend'
& '$VenvPython' -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 *>> '$BackendLog'
"@
    Start-Process powershell -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $backendCmd) | Out-Null
}

if (Test-Url "http://127.0.0.1:5173") {
    Write-Host "Frontend already running at http://127.0.0.1:5173"
} else {
    Write-Host "Starting frontend -> $FrontendLog"
    $frontendCmd = @"
`$ErrorActionPreference='Continue'
`$env:Path='$NodeDir;' + `$env:Path
`$env:npm_config_cache='$NpmCache'
Set-Location '$Frontend'
& '$Npm' run dev -- --host 127.0.0.1 --port 5173 --strictPort *>> '$FrontendLog'
"@
    Start-Process powershell -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $frontendCmd) | Out-Null
}

Write-Host "Waiting for services..."
$deadline = (Get-Date).AddSeconds(40)
$backendOk = $false
$frontendOk = $false
while ((Get-Date) -lt $deadline) {
    if (-not $backendOk) { $backendOk = Test-Url "http://127.0.0.1:8000/health" }
    if (-not $frontendOk) { $frontendOk = Test-Url "http://127.0.0.1:5173" }
    if ($backendOk -and $frontendOk) { break }
    Start-Sleep -Milliseconds 500
}

Write-Host ""
if ($backendOk) { Write-Host "OK  backend : http://127.0.0.1:8000/docs" } else { Write-Host "FAIL backend. See $BackendLog" }
if ($frontendOk) { Write-Host "OK  frontend: http://127.0.0.1:5173" } else { Write-Host "FAIL frontend. See $FrontendLog" }
Write-Host ""
Write-Host "Use http://127.0.0.1:5173  (not localhost)"

if (-not ($backendOk -and $frontendOk)) {
    if (Test-Path $BackendLog) {
        Write-Host "`n--- backend.log (tail) ---"
        Get-Content $BackendLog -Tail 30
    }
    if (Test-Path $FrontendLog) {
        Write-Host "`n--- frontend.log (tail) ---"
        Get-Content $FrontendLog -Tail 30
    }
    exit 1
}
