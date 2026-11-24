<#
.SYNOPSIS
  Clean-environment check of a release (question Q7 of the 24 Nov 2026 regression audit).
.DESCRIPTION
  In a new, empty folder: clones the release tag from GitHub, creates a new Python virtual environment, installs
  requirements.txt, downloads the core archive of the dataset record and its .sha256 from Zenodo (or uses local copies),
  and runs scripts\verify\clean_env_check.py, which regenerates Tables 9-12/S1 (RESULTS_TABLES.md), the supplementary
  tables (PHASE_D_TABLES_en.md), Table S13 and the aggregates of Table S12 from the archived inputs and compares them with
  the archived outputs. Writes clean_env_check.json in the clone and prints a summary. Nothing outside -Work is changed.
  Run it from an account or machine on which the project was never installed.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File clean_env_check.ps1 -Tag v1.0.2 -RecordId 24487851
  powershell -ExecutionPolicy Bypass -File clean_env_check.ps1 -CoreZip D:\dl\pq-pks-raw-data-v1-core.zip -Syndromes D:\dl\e1syn
#>
param(
    [string]$Tag = 'v1.0.2',
    [string]$Repo = 'https://github.com/buihuudong19/payload-leakage-pq-pks-artifact.git',
    [string]$RecordId = '24487851',
    [string]$CoreZip = '',
    [string]$Syndromes = '',
    [string]$Work = (Join-Path $env:TEMP ('pqpks-clean-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))),
    [string]$PyLauncher = 'py',
    [string]$PyVersion = '-3.12'
)
$ErrorActionPreference = 'Stop'
function Step([string]$m) { Write-Host ''; Write-Host "== $m" -ForegroundColor Cyan }
if (Test-Path $Work) { throw "$Work already exists: choose a new folder" }
New-Item -ItemType Directory -Path $Work | Out-Null
Step "clone $Repo at $Tag"
& git clone --depth 1 --branch $Tag $Repo (Join-Path $Work 'repo')
if ($LASTEXITCODE -ne 0) { throw 'git clone failed' }
$repoDir = Join-Path $Work 'repo'
Step 'new virtual environment'
& $PyLauncher $PyVersion -m venv (Join-Path $Work 'venv')
$py = Join-Path $Work 'venv\Scripts\python.exe'
& $py -m pip install --upgrade pip | Out-Null
& $py -m pip install -r (Join-Path $repoDir 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }
if (-not $CoreZip) {
    Step "download the core archive of Zenodo record $RecordId"
    $base = "https://zenodo.org/records/$RecordId/files"
    $CoreZip = Join-Path $Work 'pq-pks-raw-data-v1-core.zip'
    & curl.exe --fail --location --retry 3 -o $CoreZip "$base/pq-pks-raw-data-v1-core.zip?download=1"
    & curl.exe --fail --location --retry 3 -o "$CoreZip.sha256" "$base/pq-pks-raw-data-v1-core.zip.sha256?download=1"
    if ($LASTEXITCODE -ne 0) { throw 'download failed' }
}
Step 'regenerate and compare'
$args2 = @('scripts\verify\clean_env_check.py', '--core-zip', $CoreZip)
if ($Syndromes) { $args2 += @('--syndromes', $Syndromes) }
Push-Location $repoDir
try { & $py @args2; $rc = $LASTEXITCODE } finally { Pop-Location }
Write-Host ''
Write-Host "Log: $(Join-Path $repoDir 'clean_env_check.json')" -ForegroundColor Green
exit $rc
