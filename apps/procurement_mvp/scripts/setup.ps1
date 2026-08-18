$ErrorActionPreference = "Stop"

$MvpRoot = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $MvpRoot "backend"
$Frontend = Join-Path $MvpRoot "frontend"
$Venv = Join-Path $MvpRoot ".venv"
$CacheRoot = Join-Path $MvpRoot ".cache"
$PipCache = Join-Path $CacheRoot "pip"
$NpmCache = Join-Path $CacheRoot "npm"
$TempDir = Join-Path $CacheRoot "tmp"
$PlaywrightBrowsers = Join-Path $CacheRoot "ms-playwright"

New-Item -ItemType Directory -Force -Path $PipCache, $NpmCache, $TempDir, $PlaywrightBrowsers | Out-Null
$env:PIP_CACHE_DIR = $PipCache
$env:npm_config_cache = $NpmCache
$env:PLAYWRIGHT_BROWSERS_PATH = $PlaywrightBrowsers
$env:TEMP = $TempDir
$env:TMP = $TempDir

$PortableNodeRoot = "E:\DevTools\nodejs"
if (-not (Get-Command npm -ErrorAction SilentlyContinue) -and (Test-Path $PortableNodeRoot)) {
    $PortableNode = Get-ChildItem $PortableNodeRoot -Directory -Filter "node-*-win-x64" |
        Where-Object { Test-Path (Join-Path $_.FullName "npm.cmd") } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($PortableNode) {
        $env:Path = "$($PortableNode.FullName);$env:Path"
    }
}

if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    Write-Host "Creating isolated Python environment on D drive..."
    python -m venv $Venv
}

$VenvPython = Join-Path $Venv "Scripts\python.exe"

Write-Host "Installing backend dependencies into $Venv ..."
Push-Location $Backend
try {
    & $VenvPython -m pip install -r requirements.txt
}
finally {
    Pop-Location
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "Node.js/npm was not found. Install Node.js 20 or newer, then rerun this script."
}

Write-Host "Installing frontend dependencies..."
Push-Location $Frontend
try {
    npm install
    if ($LASTEXITCODE -ne 0) { throw "Frontend dependency installation failed." }
    npx playwright install chromium
    if ($LASTEXITCODE -ne 0) { throw "Playwright Chromium installation failed." }
}
finally {
    Pop-Location
}

& $VenvPython (Join-Path $PSScriptRoot "init_demo.py")
Write-Host "Procurement MVP setup completed."
