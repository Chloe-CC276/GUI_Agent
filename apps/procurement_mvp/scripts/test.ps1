$ErrorActionPreference = "Stop"

$MvpRoot = Split-Path -Parent $PSScriptRoot
$Backend = Join-Path $MvpRoot "backend"
$Frontend = Join-Path $MvpRoot "frontend"
$VenvPython = Join-Path $MvpRoot ".venv\Scripts\python.exe"
$env:npm_config_cache = Join-Path $MvpRoot ".cache\npm"
$env:PLAYWRIGHT_BROWSERS_PATH = Join-Path $MvpRoot ".cache\ms-playwright"

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

if (-not (Test-Path $VenvPython)) {
    throw "The D-drive Python environment is missing. Run .\scripts\setup.ps1 first."
}

Write-Host "Running backend tests..."
Push-Location $Backend
try {
    & $VenvPython -m pytest tests -q
}
finally {
    Pop-Location
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "Backend tests passed, but Node.js/npm was not found; frontend tests could not run."
}

Write-Host "Running frontend tests and production build..."
Push-Location $Frontend
try {
    npm test
    if ($LASTEXITCODE -ne 0) { throw "Frontend tests failed." }
    npm run build
    if ($LASTEXITCODE -ne 0) { throw "Frontend build failed." }
    npm run test:e2e
    if ($LASTEXITCODE -ne 0) { throw "Frontend Playwright E2E tests failed." }
}
finally {
    Pop-Location
    Remove-Item (Join-Path $Backend "e2e_s0.db*") -Force -ErrorAction SilentlyContinue
}

Write-Host "All procurement MVP checks passed."
