<#
.SYNOPSIS
  Runs the steps registered after review (lane W13: RV1-RV10) from this release folder on the measurement folder.
.DESCRIPTION
  The measurement data (BOSSbase/BOWS2/ALASKA2 images, SRM features of G4, virtual environment) live in the measurement
  folder (-Work, default ..\SOURCE-CODE-PQ-PKS). This script runs the released code of this folder with the Python of that
  folder and writes every result under -Work\results. Commit and push the folder registrations\ BEFORE running any step:
  the commit date is the date of registration.
  Steps (run them in this order; each can be rerun):
    RV1  G4 for SRM on ten fresh seeds                         1-3 h
    RV2  E1-ter image list in -Work, checked against the registered SHA-256      < 1 min
    RV3  E1-ter split                                          < 1 min
    RV4  E1-ter design: null draws with retraining (before any image)        15-40 min
    RV5  E1-ter: download 38 533 ALASKA2 images (~10 GB) with the script of -Work      2-8 h
    RV6  E1-ter: verify the images and convert them to PGM     20-60 min
    RV7  E1-ter: extract the syndromes                         1-2 h
    RV8  E1-ter: analysis with the registered rule             30-90 min
    RV9  receiver load benchmark (reference and cached receiver; concurrency 1/4/8/16)   1-2 h
    RV10 bound on epsilon_valid with G as a random oracle       < 1 min
  Each step writes its console output to -Work\results\review_logs\<step>.log.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\orchestration\run_review_steps.ps1 -Step RV1
  scripts\orchestration\run_review_steps.cmd -Step RV2        (from cmd.exe: one step per call)
#>
param(
    [Parameter(Mandatory = $true)][ValidateSet('RV1', 'RV2', 'RV3', 'RV4', 'RV5', 'RV6', 'RV7', 'RV8', 'RV9', 'RV10')][string[]]$Step,
    [string]$Work = '',
    [string]$Py = '',
    [int]$Workers = 7
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $Work) { $Work = Join-Path (Split-Path $Root -Parent) 'SOURCE-CODE-PQ-PKS' }
$Work = (Resolve-Path $Work).Path
if (-not $Py) { $Py = Join-Path $Work '.venv-confirm\Scripts\python.exe' }
if (-not (Test-Path $Py)) { throw "Python not found: $Py" }
$Logs = Join-Path $Work 'results\review_logs'
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
# Console output of Python goes through a pipe (Tee-Object); without UTF-8 mode Python encodes it with the ANSI code page
# (cp1252) and stops on any non-Latin-1 character, e.g. the Vietnamese messages of the measurement-folder scripts.
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
# Windows PowerShell reads the output of a child process with [Console]::OutputEncoding (OEM code page by default).
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }
$RegisteredIdsSha = '00565334243b9bdc7a481e2493165e8310f42cc95a5828ad2d2c2f0ffd2fb6ea'
$a2 = Join-Path $Work 'data\external\alaska2'

function Invoke-Logged([string]$Name, [string]$Exe, [string[]]$ArgList, [string]$Cwd = $Root) {
    $log = Join-Path $Logs "$Name.log"
    Write-Host "== $Name ($((Get-Date).ToString('yyyy-MM-dd HH:mm:ss')))" -ForegroundColor Cyan
    Push-Location $Cwd
    try {
        $eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        & $Exe @ArgList 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
        $rc = $LASTEXITCODE
        $ErrorActionPreference = $eap
    } finally { Pop-Location }
    if ($rc -ne 0) { throw "$Name failed (exit $rc); see $log" }
}

foreach ($s in $Step) {
    switch ($s) {
        'RV1' { Invoke-Logged $s $Py @('scripts\confirm\g4_new_seeds.py', '--g4', (Join-Path $Work 'results\g4'), '--out', (Join-Path $Work 'results\g4_new_seeds')) }
        'RV2' {
            Invoke-Logged $s $Py @('scripts\data\make_alaska2_e1ter_subset.py', '--src', (Join-Path $a2 'ALASKAv2Base_by_DSC.txt'),
                '--used', (Join-Path $a2 'e1bis_ids.txt'), (Join-Path $a2 'subset_ids.txt'), '--out', (Join-Path $a2 'e1ter_ids.txt'))
            $h = (Get-FileHash -Algorithm SHA256 (Join-Path $a2 'e1ter_ids.txt')).Hash.ToLower()
            if ($h -ne $RegisteredIdsSha) { throw "e1ter_ids.txt in $a2 differs from the registered list ($RegisteredIdsSha)" }
            Write-Host "e1ter_ids.txt equals the registered list" -ForegroundColor Green
        }
        'RV3' { Invoke-Logged $s $Py @('scripts\confirm\e1ter.py', 'split', '--work', $Work) }
        'RV4' { Invoke-Logged $s $Py @('scripts\confirm\e1ter.py', 'design', '--work', $Work, '--R', '1000', '--workers', "$Workers") }
        'RV5' { Invoke-Logged $s 'powershell' @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $Work 'scripts\data\download_alaska2.ps1'),
                    '-List', 'data\external\alaska2\e1ter_ids.txt') $Work }
        'RV6' { Invoke-Logged $s $Py @('scripts\data\verify_alaska2.py', '--list', 'data\external\alaska2\e1ter_ids.txt', '--budget', '1e9') $Work }
        'RV7' { Invoke-Logged $s $Py @('scripts\confirm\e1ter.py', 'extract', '--work', $Work, '--workers', '4') }
        'RV8' { Invoke-Logged $s $Py @('scripts\confirm\e1ter.py', 'analyze', '--work', $Work) }
        'RV9' {
            $env:PQPKS_BOSSBASE_DIR = Join-Path $Work 'data\external\bossbase'
            Invoke-Logged $s $Py @('scripts\experiments\e6_load.py', '--levels', '1', '4', '8', '16', '--images', '8', '--reps', '3',
                '--out', (Join-Path $Work 'results\e6_load'))
        }
        'RV10' { Invoke-Logged $s $Py @('scripts\theory\epsilon_valid_rom.py') }
    }
}
Write-Host "Done: $($Step -join ', '). Logs in $Logs" -ForegroundColor Green
