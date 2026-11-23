<#
.SYNOPSIS
  Status board of run_all: which steps are done, running, waiting or failed; percentage complete and time left
  (scripts\orchestration\progress.ps1); the latest log line of each running step.

.DESCRIPTION
  Refreshes every -Interval seconds (default 120) and writes the same content to results\run_status\STATUS.md for viewing
  from another machine. While open, this window keeps Windows awake (SetThreadExecutionState, power settings unchanged);
  closing it lets the machine sleep again under the previous settings.

    powershell -ExecutionPolicy Bypass -File scripts\orchestration\monitor.ps1           # continuous monitoring
    powershell -ExecutionPolicy Bypass -File scripts\orchestration\monitor.ps1 -Once     # print once and exit
#>
param([int]$Interval = 120, [switch]$Once, [switch]$DryRun)
. (Join-Path $PSScriptRoot 'common.ps1')
. (Join-Path $PSScriptRoot 'pipeline.ps1')
. (Join-Path $PSScriptRoot 'progress.ps1')

if (-not $Once) {
    $Host.UI.RawUI.WindowTitle = 'PQ-PKS - MONITOR' + $(if ($DryRun) { ' [DRY RUN]' } else { '' })
    Add-Type -Namespace PqPks -Name Power -MemberDefinition '[System.Runtime.InteropServices.DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
}
$script:ResLast = [datetime]::MinValue
$script:ResCsvLast = [datetime]::MinValue
$script:ResLine = ''
function Update-Resources {
    # Machine load, recorded so that the decision to start further steps in parallel can be checked.
    # Read at each refresh (at least 60 s apart); resources.json is written every time, resources.csv is appended after at least 5 minutes (by clock time, not by count).
    if (((Get-Date) - $script:ResLast).TotalSeconds -lt 60) { return }
    $script:ResLast = Get-Date
    try {
        $cpu = [math]::Round((Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average, 0)
        $os = Get-CimInstance Win32_OperatingSystem
        $ramFree = [math]::Round($os.FreePhysicalMemory / 1MB, 1); $ramTot = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
        $g = (& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
        $gv = if ($g) { $g.Split(',') | ForEach-Object { $_.Trim() } } else { @('', '', '', '') }
        $disk = Get-CimInstance Win32_LogicalDisk -Filter ("DeviceID='" + $Root.Substring(0, 2) + "'")
        $dFree = [math]::Round($disk.FreeSpace / 1GB, 0)
        $py = @(Get-Process -Name python -ErrorAction SilentlyContinue).Count
        $r = [ordered]@{ time = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'); cpu_load_pct = $cpu; ram_free_gb = $ramFree; ram_total_gb = $ramTot
                         gpu_util_pct = $gv[0]; gpu_mem_used_mb = $gv[1]; gpu_mem_total_mb = $gv[2]; gpu_temp_c = $gv[3]
                         disk_free_gb = $dFree; python_processes = $py; logical_cpus = [Environment]::ProcessorCount }
        [IO.File]::WriteAllText((Join-Path $StatusDir 'resources.json'), ($r | ConvertTo-Json), (New-Object Text.UTF8Encoding($false)))
        if (((Get-Date) - $script:ResCsvLast).TotalMinutes -ge 5) {
            $script:ResCsvLast = Get-Date
            $csv = Join-Path $StatusDir 'resources.csv'
            if (-not (Test-Path $csv)) { Add-Content -LiteralPath $csv -Value ((@($r.Keys) -join ',')) -Encoding UTF8 }
            Add-Content -LiteralPath $csv -Value ((@($r.Values) -join ',')) -Encoding UTF8
        }
        $script:ResLine = "CPU $cpu % ($([Environment]::ProcessorCount) threads) | free RAM $ramFree/$ramTot GB | GPU $($gv[0]) %, $($gv[1])/$($gv[2]) MB, $($gv[3]) °C | $($Root.Substring(0, 2)) free $dFree GB | $py python processes"
    } catch { $script:ResLine = "resources not readable: $($_.Exception.Message)" }
}

# Is a step alive? Sum the CPU time of every python process descending from the lane window (PID recorded in the step
# status) and divide by the time between two refreshes = number of cores in use. Steps that print no progress
# (E4 block 3, C7 between two pairs) can still be seen to be computing or stalled.
$script:ProcSnap = @()
$script:CpuPrev = @{}
function Update-ProcSnapshot {
    try { $script:ProcSnap = @(Get-CimInstance Win32_Process -Property ProcessId, ParentProcessId, Name, KernelModeTime, UserModeTime -ErrorAction Stop) }
    catch { $script:ProcSnap = @() }
}
function Get-StepCpu([string]$Id, $LanePid) {
    # returns @{ N = number of python processes; Cores = cores used since the previous measurement (or $null) }, or $null if unreadable
    if (-not $script:ProcSnap.Count -or -not $LanePid) { return $null }
    $kids = @{}
    foreach ($q in $script:ProcSnap) {
        $k = [int]$q.ParentProcessId
        if (-not $kids.ContainsKey($k)) { $kids[$k] = New-Object System.Collections.ArrayList }
        [void]$kids[$k].Add($q)
    }
    $todo = New-Object System.Collections.Queue
    $todo.Enqueue([int]$LanePid)
    $seen = @{}; $cpu = 0.0; $n = 0
    while ($todo.Count -gt 0) {
        $x = $todo.Dequeue()
        if ($seen.ContainsKey($x) -or -not $kids.ContainsKey($x)) { continue }
        $seen[$x] = $true
        foreach ($q in $kids[$x]) {
            if ([int]$q.ProcessId -eq $x) { continue }
            if ($q.Name -like 'python*') { $n++; $cpu += ([double]$q.KernelModeTime + [double]$q.UserModeTime) / 1e7 }
            $todo.Enqueue([int]$q.ProcessId)
        }
    }
    $now = Get-Date
    $cores = $null
    if ($script:CpuPrev.ContainsKey($Id)) {
        $pv = $script:CpuPrev[$Id]
        $dt = ($now - $pv.T).TotalSeconds
        if ($dt -gt 5 -and $cpu -ge $pv.Cpu) { $cores = ($cpu - $pv.Cpu) / $dt }
    }
    $script:CpuPrev[$Id] = @{ T = $now; Cpu = $cpu }
    return @{ N = $n; Cores = $cores }
}

$Label = @{ done = 'done'; skipped = 'skipped'; failed = 'FAILED'; blocked = 'BLOCKED'; running = 'RUNNING'; waiting = 'waiting'; pending = 'pending' }
$Color = @{ done = 'Green'; skipped = 'DarkGray'; failed = 'Red'; blocked = 'Red'; running = 'Yellow'; waiting = 'Gray'; pending = 'DarkGray' }

function Get-Rows {
    foreach ($s in $Steps) {
        $st = Get-StepStatus $s.Id
        $state = if ($st) { [string]$st.state } else { 'pending' }
        $note = ''
        $alive = $true
        if ($st) {
            if ($state -eq 'waiting' -and $st.waiting_for) { $note = "waiting for $($st.waiting_for)" }
            elseif (@('failed', 'blocked', 'skipped') -contains $state -and $st.reason) { $note = [string]$st.reason }
            if (@('running', 'waiting') -contains $state -and $st.pid) {
                $alive = [bool](Get-Process -Id ([int]$st.pid) -ErrorAction SilentlyContinue)
            }
            $cpuTag = ''
            if ($state -eq 'running' -and $alive -and $st.pid) {
                $c = Get-StepCpu $s.Id $st.pid
                if ($c) {
                    if ($c.N -eq 0) { $cpuTag = 'no python process found | ' }
                    elseif ($null -ne $c.Cores) {
                        $cpuTag = if ($c.Cores -lt 0.02) { 'NO CPU USE (possibly stalled) | ' } else { 'CPU {0:0.0} cores | ' -f $c.Cores }
                    }
                }
            }
            if ($state -eq 'running' -and $st.log) {
                $lp = Join-Path $Root $st.log
                if (Test-Path -LiteralPath $lp) {
                    $last = Get-Content -LiteralPath $lp -Tail 1 -Encoding UTF8 -ErrorAction SilentlyContinue
                    if ($last) { $note = ([string]$last).Trim() }
                }
            }
            $note = $cpuTag + $note
        }
        if (-not $alive) { $note = "lane window closed: rerun run_all.ps1 | $note" }
        if ($note.Length -gt 90) { $note = $note.Substring(0, 87) + '...' }
        $pr = Get-StepProgress $s $st
        [pscustomobject]@{
            Id = $s.Id; Lane = $s.Lane; Name = $s.Name; State = $state; Alive = $alive; Est = $s.Est
            Start = $(if ($st -and $st.start) { ([string]$st.start).Substring(5, 11) } else { '' })
            Dur = $(if ($st -and $st.start) { Format-Duration ([string]$st.start) ([string]$st.end) } else { '' })
            Pct = (Format-Pct $pr); Bar = (Format-Bar $pr); Left = (Format-Left $pr); P = $pr.Pct
            Note = $note
        }
    }
}

while ($true) {
    if (-not $Once) { [void][PqPks.Power]::SetThreadExecutionState([uint32]2147483649) }   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    Update-ProcSnapshot
    $rows = @(Get-Rows)
    $n = $rows.Count
    $cnt = @{}
    foreach ($r in $rows) { $cnt[$r.State] = 1 + [int]$cnt[$r.State] }
    $now = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    $head = "PQ-PKS run_all  $now   done $([int]$cnt['done'] + [int]$cnt['skipped'])/$n | running $([int]$cnt['running']) | waiting $([int]$cnt['waiting'] + [int]$cnt['pending']) | failed/blocked $([int]$cnt['failed'] + [int]$cnt['blocked'])"
    if (-not $Once) { Clear-Host }
    Write-Host $head -ForegroundColor Cyan
    if (-not $Once -and -not $DryRun) { Update-Resources; if ($script:ResLine) { Write-Host ("Machine: " + $script:ResLine) -ForegroundColor DarkCyan } }
    Write-Host ('-' * 160)
    Write-Host ('{0,-4} {1,-3} {2,-44} {3,-10} {4,6} {5,-10} {6,-9} {7,-11} {8,6}  {9}' -f 'Step', 'Lane', 'Task', 'State', '% done', '', 'Left', 'Started', 'Elapsed', 'Note / latest log line')
    foreach ($r in $rows) {
        $nm = if ($r.Name.Length -gt 44) { $r.Name.Substring(0, 43) + '…' } else { $r.Name }
        $c = $Color[$r.State]
        if (-not $r.Alive) { $c = 'Magenta' }
        Write-Host ('{0,-4} {1,-3} {2,-44} {3,-10} {4,6} {5,-10} {6,-9} {7,-11} {8,6}  {9}' -f $r.Id, $r.Lane, $nm, $Label[$r.State], $r.Pct, $r.Bar, $r.Left, $r.Start, $r.Dur, $r.Note) -ForegroundColor $c
    }
    $md = @("# run_all status", "", $head, "", $(if ($script:ResLine) { "Machine: " + $script:ResLine } else { '' }), "", "| Step | Lane | Task | State | % done | Left | Started | Elapsed | Estimate | Note |", "|---|---|---|---|---|---|---|---|---|---|")
    foreach ($r in $rows) {
        $md += ('| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} |' -f $r.Id, $r.Lane, $r.Name, $Label[$r.State], $r.Pct, $r.Left, $r.Start, $r.Dur, $r.Est, ($r.Note -replace '\|', '/'))
    }
    $md += @('', '% done: read from the step log (C4 sums 6 workers; SRNet by iteration; C7 by trained pairs; E4 by block). "~": estimate without a progress line. Left: SRNet from it/s, C7 from the mean time of the 4 most recent pairs, E4 block 3 from Table 12 scaled to the speed of this machine. "CPU x cores": cores used by the python processes of the step since the previous refresh; "NO CPU USE" suggests a stalled step.')
    try { $md | Set-Content -LiteralPath (Join-Path $StatusDir 'STATUS.md') -Encoding UTF8 } catch { }
    $left = @($rows | Where-Object { -not ($TerminalStates -contains $_.State) }).Count
    if ($Once) { break }
    if ($left -eq 0) {
        Write-Host ''
        Write-Host 'Every step has ended. Collected results: results\manuscript_inputs\INDEX.md' -ForegroundColor Cyan
        break
    }
    Write-Host '% done is read from the step log. "~": estimate. Left: SRNet from it/s, C7 from the 4 most recent pairs, E4 block 3 from Table 12. "CPU x cores": the step is computing; "NO CPU USE": possibly stalled.' -ForegroundColor DarkGray
    Write-Host ''
    Write-Host "Refresh in $Interval s. This window keeps the machine awake; closing it does not stop the lanes." -ForegroundColor DarkGray
    Start-Sleep -Seconds $Interval
}
