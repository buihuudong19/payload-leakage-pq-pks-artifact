<#
.SYNOPSIS
  Runs the steps of one lane (W1..W12) of scripts\orchestration\pipeline.ps1 in order. run_all.ps1 opens each automatic lane
  in its own PowerShell window; the manual lanes W10, W11 and W12 are started with run_lane.ps1 -Lane W10 (or W11, W12).

.DESCRIPTION
  For each step: skip it if the status file records done/skipped; wait for the dependencies (checked every 30 s); evaluate
  the When condition; run the command, writing its output to results\logs\<step>.log and to the console; check the exit code
  and Check; record the status. A step interrupted midway (window closed, power loss) stays in state running; rerunning
  run_all.ps1 runs it again. The long scripts (C1, C4, C5, 3a, 3e, SRNet) resume from the part already finished.
#>
param(
    [Parameter(Mandatory = $true)][ValidateSet('W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'W10', 'W11', 'W12', 'W13')][string]$Lane,
    [string[]]$Only = @(),
    [switch]$DryRun
)
# -Only: run only the listed steps of the lane, in a separate window, without the implicit dependency on the previous step
# of the lane (only the Deps declared in pipeline.ps1 are kept). Useful when the inputs of a step are ready while the lane is
# busy with an unrelated step. The original lane window then waits for the step (if still running) or skips it (if done), so nothing runs twice.
$Only = @($Only | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
. (Join-Path $PSScriptRoot 'common.ps1')
. (Join-Path $PSScriptRoot 'pipeline.ps1')

$Host.UI.RawUI.WindowTitle = "PQ-PKS $Lane - $($LaneTitleAll[$Lane])" + $(if ($Only.Count) { " [only $($Only -join ', ')]" } else { '' }) + $(if ($DryRun) { ' [DRY RUN]' } else { '' })
Write-Host "=== Lane $Lane : $($LaneTitleAll[$Lane]) ===" -ForegroundColor Cyan
if ($DryRun) { Write-Host 'DRY RUN: no command is executed, status written to results\run_status_dryrun\' -ForegroundColor Yellow }

$mine = @($Steps | Where-Object { $_.Lane -eq $Lane })
if ($Only.Count) {
    $miss = @($Only | Where-Object { $o = $_; -not ($mine | Where-Object { $_.Id -eq $o }) })
    if ($miss.Count) { Write-Host "Steps not in lane ${Lane}: $($miss -join ', ')" -ForegroundColor Red; exit 1 }
    $mine = @($mine | Where-Object { $Only -contains $_.Id })
    Write-Host "Running only: $($Only -join ', ') (no dependency on the previous step of the lane)" -ForegroundColor Cyan
}
$prev = $null
$prevSoft = $false
foreach ($s in $mine) {
    $id = $s.Id
    $st = Get-StepStatus $id
    # The step is running in another window (process alive): wait for it to end, never run it twice.
    $other = $false
    while ($st -and [string]$st.state -eq 'running' -and $st.pid -and [int]$st.pid -ne $PID -and
           (Get-Process -Id ([int]$st.pid) -ErrorAction SilentlyContinue)) {
        if (-not $other) {
            Write-Host ("[{0}] {1}: running in another window (PID {2}), waiting for it  ({3})" -f $id, $s.Name, $st.pid, (Get-Date).ToString('HH:mm')) -ForegroundColor DarkYellow
            $other = $true
        }
        Start-Sleep -Seconds $(if ($DryRun) { 2 } else { 30 })
        $st = Get-StepStatus $id
    }
    if ($other -and $st -and @('failed', 'blocked') -contains [string]$st.state) {
        Write-Host "[$id] $($s.Name): the other window ended with state $($st.state), not rerun" -ForegroundColor Red
        $prev = $id
        continue
    }
    if ($st -and ($OkStates -contains [string]$st.state)) {
        Write-Host "[$id] $($s.Name): already $($st.state), skipped" -ForegroundColor DarkGray
        $prev = $id
        continue
    }
    $deps = @()
    if ($prev) { if ($s.Soft) { $deps += "~$prev" } else { $deps += $prev } }
    $deps += @($s.Deps)
    $base = @{ id = $id; lane = $Lane; name = $s.Name; est = $s.Est; out = $s.Out; pid = $PID; start = $null; end = $null;
               exit = $null; reason = $null; log = $null }
    Set-StepStatus $id ($base + @{ state = 'waiting'; waiting_for = ($deps -join ', ') })
    $blocked = $null
    $shown = ''
    while ($true) {
        $pending = @(); $bad = @()
        foreach ($d in $deps) {
            $r = Resolve-Dep $d
            if ($r -eq 'wait') { $pending += $d } elseif ($r -eq 'bad') { $bad += $d }
        }
        if ($bad.Count -gt 0) { $blocked = ($bad -join ', '); break }
        if ($pending.Count -eq 0) { break }
        $w = ($pending -join ', ')
        if ($w -ne $shown) {
            Set-StepStatus $id @{ state = 'waiting'; waiting_for = $w }
            Write-Host ("[{0}] {1}: waiting for {2}  ({3})" -f $id, $s.Name, $w, (Get-Date).ToString('HH:mm')) -ForegroundColor DarkYellow
            $shown = $w
        }
        Start-Sleep -Seconds $(if ($DryRun) { 2 } else { 30 })
    }
    if ($blocked) {
        Set-StepStatus $id @{ state = 'blocked'; waiting_for = $null; reason = "dependency not completed: $blocked" }
        Write-Host "[$id] $($s.Name): BLOCKED by $blocked" -ForegroundColor Red
        $prev = $id
        continue
    }
    if ($s.When -and -not $DryRun -and -not (& $s.When)) {
        Set-StepStatus $id @{ state = 'skipped'; waiting_for = $null; reason = $s.WhenText; end = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss') }
        Write-Host "[$id] $($s.Name): skipped ($($s.WhenText))" -ForegroundColor DarkGray
        $prev = $id
        continue
    }

    $log = Join-Path $LogDir ("{0}.log" -f $id)
    $t0 = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    Set-StepStatus $id @{ state = 'running'; waiting_for = $null; start = $t0; log = ($log.Substring($Root.Length + 1)) }
    switch ($s.Kind) {
        'py'  { $exe = $Py;  $argv = @($s.Args) }
        'pyg' { $exe = $PyG; $argv = @($s.Args) }
        'ps'  { $exe = 'powershell.exe'; $argv = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File') + @($s.Args) }
    }
    $cmdline = "$exe " + ($argv -join ' ')
    Write-Host ''
    Write-Host ("[{0}] {1}" -f $id, $s.Name) -ForegroundColor Green
    Write-Host ("    {0}    started {1}, estimate {2}" -f $cmdline, $t0, $s.Est) -ForegroundColor Green
    Add-Content -LiteralPath $log -Value ("===== {0}  {1}" -f $t0, $cmdline) -Encoding UTF8
    if ($DryRun) {
        Add-Content -LiteralPath $log -Value 'DRY RUN: command not executed' -Encoding UTF8
        Start-Sleep -Seconds 3
        $code = 0
    } else {
        & $exe @argv 2>&1 | ForEach-Object {
            $line = "$_"
            Write-Host $line
            Add-Content -LiteralPath $log -Value $line -Encoding UTF8
        }
        $code = $LASTEXITCODE
    }
    $err = $null
    if ($code -ne 0) { $err = "exit code $code" }
    elseif ($s.Check -and -not $DryRun) { $err = & $s.Check }
    $t1 = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    if ($err) {
        Set-StepStatus $id @{ state = 'failed'; end = $t1; exit = $code; reason = [string]$err }
        Add-Content -LiteralPath $log -Value "===== FAILED: $err" -Encoding UTF8
        Write-Host "[$id] FAILED: $err. Log: $log" -ForegroundColor Red
    } else {
        Set-StepStatus $id @{ state = 'done'; end = $t1; exit = $code }
        Write-Host ("[{0}] done after {1}" -f $id, (Format-Duration $t0 $t1)) -ForegroundColor Green
    }
    $prev = $id
}
Write-Host ''
Write-Host "=== Lane $Lane has gone through all its steps ($((Get-Date).ToString('yyyy-MM-dd HH:mm'))). This window can be closed. ===" -ForegroundColor Cyan
