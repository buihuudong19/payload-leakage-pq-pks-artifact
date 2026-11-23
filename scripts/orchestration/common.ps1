# Shared code of run_all.ps1, run_lane.ps1 and monitor.ps1 (dot-sourced, not run on its own).
# Sets up the environment as described in docs\reproduction.md and provides the functions that read and write the status of each step.
# The status of each step is a JSON file in results\run_status\ (dry run -DryRun: results\run_status_dryrun\).

$ErrorActionPreference = 'Continue'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root

if ($DryRun) {
    $StatusDir = Join-Path $Root 'results\run_status_dryrun'
    $LogDir = Join-Path $Root 'results\logs_dryrun'
} else {
    $StatusDir = Join-Path $Root 'results\run_status'
    $LogDir = Join-Path $Root 'results\logs'
}
New-Item -ItemType Directory -Force $StatusDir, $LogDir | Out-Null

Remove-Item Env:PQPKS_ENS_FIXED, Env:PQPKS_SPLIT -ErrorAction SilentlyContinue
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$Py = Join-Path $Root '.venv-confirm\Scripts\python.exe'
$PyG = Join-Path $Root '.venv-gpu\Scripts\python.exe'

$OkStates = @('done', 'skipped')
$TerminalStates = @('done', 'skipped', 'failed', 'blocked')

function Read-JsonFile([string]$Path) {
    if (-not [IO.Path]::IsPathRooted($Path)) { $Path = Join-Path $Root $Path }
    for ($i = 0; $i -lt 5; $i++) {
        if (-not (Test-Path -LiteralPath $Path)) { return $null }
        try { return (Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json) }
        catch { Start-Sleep -Milliseconds 300 }          # file being replaced: read again
    }
    return $null
}

function Get-StepStatus([string]$Id) {
    return Read-JsonFile (Join-Path $StatusDir "$Id.json")
}

function Set-StepStatus([string]$Id, [hashtable]$Fields) {
    $p = Join-Path $StatusDir "$Id.json"
    $cur = [ordered]@{}
    $old = Get-StepStatus $Id
    if ($old) { foreach ($pr in $old.PSObject.Properties) { $cur[$pr.Name] = $pr.Value } }
    foreach ($k in $Fields.Keys) { $cur[$k] = $Fields[$k] }
    $cur['updated'] = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    $tmp = "$p.$PID.tmp"
    ($cur | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath $tmp -Encoding UTF8
    Move-Item -LiteralPath $tmp -Destination $p -Force
}

function Get-StateOf([string]$Id) {
    $st = Get-StepStatus $Id
    if ($st) { return [string]$st.state } else { return 'pending' }
}

# A dependency evaluates to 'ok', 'wait' or 'bad'.
#   X          must be done or skipped by rule; 'bad' if X failed or is blocked
#   ~X         only wait for X to end, whether it succeeded or not
#   serial:X   as X, but only when results\gpu_bench\decision.json decides to run a single GPU process at a time
#   file:P     wait until file P (relative to the project directory) exists; for a step that needs only a file another step writes at its start
function Resolve-Dep([string]$Dep) {
    if ($Dep.StartsWith('file:')) {
        if ($DryRun -or (Test-Path -LiteralPath (Join-Path $Root $Dep.Substring(5)))) { return 'ok' } else { return 'wait' }
    }
    if ($Dep.StartsWith('~')) {
        if ($TerminalStates -contains (Get-StateOf $Dep.Substring(1))) { return 'ok' } else { return 'wait' }
    }
    if ($Dep.StartsWith('serial:')) {
        $x = $Dep.Substring(7)
        if (-not $DryRun) {
            $d = Read-JsonFile 'results\gpu_bench\decision.json'
            if (-not $d) {                                   # no decision yet: wait for G2b; if G2b failed, run sequentially
                if (-not ($TerminalStates -contains (Get-StateOf 'G2b'))) { return 'wait' }
            } elseif ($d.parallel) { return 'ok' }
        }
        $Dep = $x
    }
    $s = Get-StateOf $Dep
    if ($OkStates -contains $s) { return 'ok' }
    if (@('failed', 'blocked') -contains $s) { return 'bad' }
    return 'wait'
}

function Format-Duration([string]$Start, [string]$End) {
    if (-not $Start) { return '' }
    $a = [datetime]::ParseExact($Start, 'yyyy-MM-dd HH:mm:ss', $null)
    if ($End) { $b = [datetime]::ParseExact($End, 'yyyy-MM-dd HH:mm:ss', $null) } else { $b = Get-Date }
    $t = $b - $a
    return ('{0}:{1:00}' -f [int][math]::Floor($t.TotalHours), $t.Minutes)
}
