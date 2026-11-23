<#
.SYNOPSIS
  Runs the whole confirmatory step of PQ-PKS on a Windows desktop: opens one window per automatic lane and a monitor window.

.DESCRIPTION
  Steps, lanes and dependencies: scripts\orchestration\pipeline.ps1 (see docs\reproduction.md).
  Status of each step: results\run_status\<step>.json and results\run_status\STATUS.md. Logs: results\logs\<step>.log.
  Summary results are collected in results\manuscript_inputs\ (step R1).

  Rerun run_all.ps1 after an interruption: finished steps are skipped, an unfinished step is run again (long scripts resume).

.PARAMETER Reset
  Delete the status of the previous run (only the files in results\run_status\; data and results are not touched).
.PARAMETER DryRun
  Test the orchestration: opens every window and walks through every step in dependency order without running any command.
  Status is written to results\run_status_dryrun\.
.PARAMETER NoMonitor
  Do not open the monitor window.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -DryRun     # test the orchestration, about 2 minutes
  powershell -ExecutionPolicy Bypass -File .\run_all.ps1 -Reset      # full run from the beginning
  powershell -ExecutionPolicy Bypass -File .\run_all.ps1             # resume after an interruption
#>
param([switch]$Reset, [switch]$DryRun, [switch]$NoMonitor)
. (Join-Path $PSScriptRoot 'scripts\orchestration\common.ps1')
. (Join-Path $PSScriptRoot 'scripts\orchestration\pipeline.ps1')

Write-Host 'PQ-PKS run_all' -ForegroundColor Cyan
Write-Host "Directory: $Root"

# 1. Never run twice at the same time: a running/waiting step whose lane window is alive means another run is in progress.
$live = @()
foreach ($s in $Steps) {
    $st = Get-StepStatus $s.Id
    if ($st -and @('running', 'waiting') -contains [string]$st.state -and $st.pid) {
        if (Get-Process -Id ([int]$st.pid) -ErrorAction SilentlyContinue) { $live += $s.Id }
    }
}
if ($live.Count -gt 0) {
    Write-Host "Another run is in progress (steps $($live -join ', ') still have a lane window). Close those windows first." -ForegroundColor Red
    exit 1
}

# 2. Preconditions
$bad = @()
foreach ($d in @('data\external\bossbase', 'data\external\bows2')) {
    if (-not (Test-Path (Join-Path $Root $d))) { $bad += "missing $d" }
}
$drive = Get-PSDrive -Name (Split-Path -Qualifier $Root).TrimEnd(':')
$freeGB = [math]::Round($drive.Free / 1GB)
if ($freeGB -lt 250) { $bad += "drive $($drive.Name): only $freeGB GB free, about 250 GB needed (5a 110 GB, memmaps 58 GB, SRNet 30 GB)" }
if ($bad.Count -gt 0 -and -not $DryRun) {
    $bad | ForEach-Object { Write-Host "Cannot run: $_" -ForegroundColor Red }
    exit 1
}
Write-Host "Drive $($drive.Name): $freeGB GB free"

# 3. Status
if ($Reset) {
    Get-ChildItem -LiteralPath $StatusDir -File -ErrorAction SilentlyContinue | Remove-Item -Force
    Write-Host "Previous status deleted in $StatusDir" -ForegroundColor Yellow
}
$frozen = Read-JsonFile 'configs\profile_frozen_v1.json'
$run = [ordered]@{
    started = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'); computer = $env:COMPUTERNAME; dry_run = [bool]$DryRun
    steps = $Steps.Count; frozen_date = $(if ($frozen) { $frozen.date } else { $null })
}
($run | ConvertTo-Json) | Set-Content -LiteralPath (Join-Path $StatusDir 'run.json') -Encoding UTF8

# 4. Open the windows
$extra = @()
if ($DryRun) { $extra = @('-DryRun') }
foreach ($lane in $LaneTitle.Keys) {
    $a = @('-NoExit', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$(Join-Path $PSScriptRoot 'scripts\orchestration\run_lane.ps1')`"", '-Lane', $lane) + $extra
    Start-Process -FilePath 'powershell.exe' -ArgumentList $a -WorkingDirectory $Root
    Write-Host "Lane $lane opened: $($LaneTitle[$lane])" -ForegroundColor Green
    Start-Sleep -Milliseconds 700
}
if (-not $NoMonitor) {
    $a = @('-NoExit', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$(Join-Path $PSScriptRoot 'scripts\orchestration\monitor.ps1')`"") + $extra
    Start-Process -FilePath 'powershell.exe' -ArgumentList $a -WorkingDirectory $Root
    Write-Host 'Monitor window opened (keeps the machine awake during the run)' -ForegroundColor Green
}
Write-Host ''
Write-Host "Monitor: window 'PQ-PKS - MONITOR', or results\run_status\STATUS.md. Logs of each step: results\logs\." -ForegroundColor Cyan
