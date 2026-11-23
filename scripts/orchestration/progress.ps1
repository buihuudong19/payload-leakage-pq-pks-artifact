# Estimate the completion percentage of a step from its log (dot-sourced; used by monitor.ps1). Read-only.
#
# Parses the actual log format of each script:
#   C4 (screen_run.py)          "[w3] 2761/3333 53341s"         sums the latest progress of every worker
#   SRNet (srnet_train.py)      "iteration 37700/500000 ... 0.71 it/s"
#                               target = "this run up to N" if present (--until), origin = branch/resume iteration of the first run
#   C7 (analyze_confirm.py)     "trained <alpha> <system> 3 reps in Ns" (training, 18 pairs), then "<alpha> <system> {" (aggregation, 18 pairs)
#                               remaining = untrained pairs x mean time of the 4 most recent pairs
#   E4 (e4_functional.py)       "normal {...} Ns", "tamper {...} Ns" (seconds from the start); block 3 prints no progress: estimated as
#                               number of cover images x 7.309 s (Table 12, reference machine) x the speed factor of this machine measured in block 1
#   C1, C2, C5, C6, 3a, 3e      "k/N after" or "k / N"
#   Steps without progress output (1, 2, G1, E4...) are estimated from elapsed time / estimate (middle of the Est range), at most 95 %,
#   marked Approx = $true so that the table shows "~".
# Otherwise: SRNet from the it/s rate in the log; other steps from elapsed time x (1 - p) / p.
# A time-estimated step that has run longer than its estimate is reported as "over estimate" (Over = $true).

$script:ProgCache = @{}

function Get-EstHours([string]$Est) {
    # '10.5-17.5 h' -> 14; '~11 h' -> 11; '< 1 min' -> 0.0167; '~8 h (~16 in parallel)' -> 8
    if (-not $Est) { return $null }
    $m = [regex]::Match($Est, '([0-9]+(?:[.,][0-9]+)?)(?:\s*-\s*([0-9]+(?:[.,][0-9]+)?))?\s*(h|min)\b')
    if (-not $m.Success) { return $null }
    $a = [double]($m.Groups[1].Value -replace ',', '.')
    $b = if ($m.Groups[2].Success) { [double]($m.Groups[2].Value -replace ',', '.') } else { $a }
    $h = ($a + $b) / 2
    if ($m.Groups[3].Value -eq 'min') { $h = $h / 60 }
    return $h
}

function Get-CurrentBlock([string[]]$Lines) {
    # lines of the current run: after the last "=====" line (run_lane writes one such line each time a step starts).
    # Emits the lines one by one to the pipeline (no array wrapping); callers use @(...)
    $k = -1
    for ($i = $Lines.Count - 1; $i -ge 0; $i--) { if ($Lines[$i].TrimStart([char]0xFEFF).StartsWith('=====')) { $k = $i; break } }
    if ($k -lt 0) { return $Lines }
    if ($k + 1 -ge $Lines.Count) { return }
    return $Lines[($k + 1)..($Lines.Count - 1)]
}

function Get-SrnetFrame([string]$Id, [string]$LogPath, [string]$Start) {
    # origin and target of an SRNet step; the whole log is read once per step run and cached
    $key = "$Id|$Start"
    if ($script:ProgCache.ContainsKey($key)) { return $script:ProgCache[$key] }
    $all = @(Get-Content -LiteralPath $LogPath -Encoding UTF8 -ErrorAction SilentlyContinue)
    $base = 0
    foreach ($l in $all) {                       # first run of the step: branch/resume before the first progress line
        if ($l -match 'iteration [0-9]+/[0-9]+ loss') { break }
        if ($l -match '(resuming from iteration|branching from \S+ at iteration) ([0-9]+)') { $base = [int]$Matches[2]; break }
    }
    $target = $null
    foreach ($l in (Get-CurrentBlock $all)) {
        if ($l -match 'this run up to ([0-9]+)') { $target = [int]$Matches[1]; break }
    }
    $f = @{ base = $base; target = $target }
    $script:ProgCache[$key] = $f
    return $f
}

function Get-StepProgress($Step, $Status) {
    # returns @{ Pct = 0..100 or $null; Approx = bool; LeftH = hours left or $null }
    $none = @{ Pct = $null; Approx = $false; LeftH = $null }
    if (-not $Status) { return $none }
    $state = [string]$Status.state
    if ($state -eq 'done') { return @{ Pct = 100.0; Approx = $false; LeftH = 0 } }
    if ($state -ne 'running') { return $none }
    $elapsedH = $null
    try { $elapsedH = ((Get-Date) - [datetime]::ParseExact([string]$Status.start, 'yyyy-MM-dd HH:mm:ss', $null)).TotalHours } catch { }
    $p = $null; $left = $null; $noElapsed = $false; $approx = $false; $over = $false
    try {
        $lp = if ($Status.log) { Join-Path $Root ([string]$Status.log) } else { $null }
        $tail = @()
        if ($lp -and (Test-Path -LiteralPath $lp)) { $tail = @(Get-Content -LiteralPath $lp -Tail 400 -Encoding UTF8 -ErrorAction SilentlyContinue) }
        $blk = @(Get-CurrentBlock $tail)

        # C4: several workers "[wN] k/T"
        # the speed of each worker comes from its own two latest lines (after a resume, the count includes images finished earlier)
        $wk = @{}; $wprev = @{}
        foreach ($l in $blk) {
            if ($l -match '^\[w([0-9]+)\] ([0-9]+)/([0-9]+) ([0-9]+)s') {
                if ($wk.ContainsKey($Matches[1])) { $wprev[$Matches[1]] = $wk[$Matches[1]] }
                $wk[$Matches[1]] = @([double]$Matches[2], [double]$Matches[3], [double]$Matches[4])
            }
        }
        if ($wk.Count -gt 0) {
            $d = 0.0; $t = 0.0; $lw = 0.0; $known = $true
            foreach ($k in $wk.Keys) {
                $v = $wk[$k]; $d += $v[0]; $t += $v[1]
                if ($wprev.ContainsKey($k) -and $v[2] -gt $wprev[$k][2] -and $v[0] -gt $wprev[$k][0]) {
                    $rate = ($v[0] - $wprev[$k][0]) / ($v[2] - $wprev[$k][2])
                    $lw = [math]::Max($lw, ($v[1] - $v[0]) / $rate / 3600)
                } else { $known = $false }
            }
            if ($t -gt 0) { $p = $d / $t }
            if ($known) { $left = $lw }
            $noElapsed = $true          # no estimate from the start time: the count includes images finished before the resume
        }

        # SRNet: "iteration X/Y ... R it/s"
        if ($null -eq $p) {
            $last = $null
            foreach ($l in $blk) { if ($l -match 'iteration ([0-9]+)/([0-9]+) loss .* ([0-9]+\.[0-9]+) it/s') { $last = $Matches.Clone() } }
            if ($last) {
                $f = Get-SrnetFrame $Step.Id $lp ([string]$Status.start)
                $x = [double]$last[1]
                $target = if ($f.target) { [double]$f.target } else { [double]$last[2] }
                if ($target -gt $f.base) { $p = ($x - $f.base) / ($target - $f.base) }
                $rate = [double]$last[3]
                if ($rate -gt 0) { $left = [math]::Max(0, $target - $x) / $rate / 3600 }
            }
        }

        # C7: 18 pairs; training ("trained ... in Ns" lines) takes almost all the time, aggregation ("<alpha> <system> {" lines) is short
        # (also used for D4, D7, D10 of phase D; the number of pairs comes from the Cells property in pipeline.ps1)
        if ($null -eq $p -and ($Step.Id -eq 'C7' -or $Step.Cells)) {
            $NC = if ($Step.Cells) { [int]$Step.Cells } else { 18 }
            $durs = @(); $nres = 0
            foreach ($l in $blk) {
                if ($l -match '^trained \S+ \S+ [0-9]+ reps in ([0-9]+)s') { $durs += [double]$Matches[1] }
                elseif ($l -match '^\S+ \S+ \{') { $nres++ }
            }
            $nt = $durs.Count
            if ($nt -lt $NC) {
                $p = 0.95 * $nt / $NC
                if ($nt -gt 0) {
                    $k = [math]::Min(4, $nt)
                    $mean = ($durs[($nt - $k)..($nt - 1)] | Measure-Object -Average).Average
                    $left = ($NC - $nt) * $mean / 3600
                }
            } else {
                $p = 0.95 + 0.05 * $nres / $NC
                $trainH = ($durs | Measure-Object -Sum).Sum / 3600
                if ($elapsedH -and $nres -gt 0) { $left = [math]::Max(0, $elapsedH - $trainH) / $nres * ($NC - $nres) }
            }
            $noElapsed = $true
        }

        # E4: three blocks; block 3 (false alarms) prints no progress
        if ($null -eq $p -and $Step.Id -eq 'E4') {
            $nNorm = $null; $nCov = $null; $tn = $null; $tt = $null
            foreach ($l in $tail) {
                if ($l.TrimStart([char]0xFEFF).StartsWith('=====') -and $l -match '--normal ([0-9]+)') {
                    $nNorm = [double]$Matches[1]
                    if ($l -match '--cover-images ([0-9]+)') { $nCov = [double]$Matches[1] }
                }
            }
            foreach ($l in $blk) {
                if ($l -match '^normal \{.*\} ([0-9]+)s') { $tn = [double]$Matches[1] }
                elseif ($l -match '^tamper \{.*\} ([0-9]+)s') { $tt = [double]$Matches[1] }
            }
            if ($tn -and $tt -and $nNorm -and $nCov -and $elapsedH) {
                # speed factor: send + receive time of one image in block 1 relative to the reference machine of Table 12 (0.722 s + 0.016 s)
                $speed = ($tn / $nNorm) / 0.738
                $total = $tt + $nCov * 7.309 * $speed
                $el = $elapsedH * 3600
                $p = [math]::Min(0.99, $el / $total)
                $left = [math]::Max(0, $total - $el) / 3600
                $over = $el -gt $total
                $approx = $true; $noElapsed = $true
            }
        }

        # Chunks: "k/N after" or "k / N"
        if ($null -eq $p) {
            $last = $null
            foreach ($l in $blk) {
                if ($l -match '([0-9]+)/([0-9]+) after ') { $last = @([double]$Matches[1], [double]$Matches[2]) }
                elseif ($l -match 'Processed ([0-9]+)/([0-9]+)') { $last = @([double]$Matches[1], [double]$Matches[2]) }
                elseif ($l -match '^([0-9]+) / ([0-9]+)\s*$') { $last = @([double]$Matches[1], [double]$Matches[2]) }
            }
            if ($last -and $last[1] -gt 0) { $p = $last[0] / $last[1] }
        }
    } catch { $p = $null }

    if ($null -ne $p) {
        $p = [math]::Min(1.0, [math]::Max(0.0, $p))
        if ($null -eq $left -and -not $noElapsed -and $elapsedH -and $p -ge 0.005) { $left = $elapsedH * (1 - $p) / $p }
        return @{ Pct = 100 * $p; Approx = $approx; LeftH = $left; Over = $over }
    }
    # No progress line: estimate from elapsed time
    $estH = Get-EstHours ([string]$Step.Est)
    if ($estH -and $elapsedH) {
        $q = [math]::Min(0.95, $elapsedH / $estH)
        return @{ Pct = 100 * $q; Approx = $true; LeftH = [math]::Max(0, $estH - $elapsedH); Over = ($elapsedH -gt $estH) }
    }
    return $none
}

function Format-Pct($Pr) {
    if ($null -eq $Pr.Pct) { return '' }
    $s = '{0:0.0}%' -f $Pr.Pct
    if ($Pr.Pct -ge 100) { $s = '100%' }
    if ($Pr.Approx) { $s = '≈' + ('{0:0}%' -f $Pr.Pct) }
    return $s
}

function Format-Bar($Pr, [int]$W = 10) {
    if ($null -eq $Pr.Pct) { return (' ' * $W) }
    $k = [int][math]::Floor($Pr.Pct / 100 * $W)
    return ('#' * $k) + ('.' * ($W - $k))
}

function Format-Left($Pr) {
    if ($Pr.Pct -ge 100) { return '' }
    if ($Pr.Over) { return 'over estimate' }
    if ($null -eq $Pr.LeftH) { return '' }
    $h = [double]$Pr.LeftH
    $s = if ($h -ge 48) { '{0:0.0} d' -f ($h / 24) } elseif ($h -ge 1) { '{0}h{1:00}' -f [int][math]::Floor($h), [int][math]::Floor(($h - [math]::Floor($h)) * 60) } else { '{0} min' -f [int][math]::Ceiling($h * 60) }
    if ($Pr.Approx) { $s = '≈' + $s }
    return $s
}
