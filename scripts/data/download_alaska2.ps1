<#
.SYNOPSIS
  Downloads the ALASKA2 subset (512x512 grayscale, uncompressed TIFF) for experiment E7.

.DESCRIPTION
  List of the 10 000 images: data\external\alaska2\subset_ids.txt (stratified by camera model, seed 20261029, generated with
  scripts\data\make_alaska2_subset.py). Source: https://alaska.utt.fr (licence CC BY-NC-ND, non-commercial use only; cite ALASKA).
  Each image is about 262 570 bytes; about 2.6 GB in total.
  The ALASKA site asks users not to overload the server: by default 4 images are downloaded in parallel, in batches of 200.
  Can be rerun: images already downloaded with the full size are skipped.

.EXAMPLE
  # From the repository root, in a normal PowerShell:
  powershell -ExecutionPolicy Bypass -File scripts\data\download_alaska2.ps1
#>
param(
    [int]$Parallel = 4,
    [int]$BatchSize = 200,
    [string]$List = '',     # default: the E7 list; E1-bis: data\external\alaska2\e1bis_ids.txt
    [int]$Passes = 3        # number of passes over the images still missing (transient network errors leave partial files)
)
$ErrorActionPreference = 'Stop'
$Root    = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $List) { $List = Join-Path $Root 'data\external\alaska2\subset_ids.txt' } elseif (-not [IO.Path]::IsPathRooted($List)) { $List = Join-Path $Root $List }
$OutDir  = Join-Path $Root 'data\external\raw\alaska2_tiff512'
$BaseUrl = 'https://alaska.utt.fr/DATASETS/ALASKA_v2_TIFF_512_GrayScale'
$Expect  = @(262570, 262314)   # uncompressed 8-bit 512x512 TIFF: 32 strips (262 570 B) or 1 strip (262 314 B, seen in 21 images); step X2 decodes every image to check
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Log = Join-Path $OutDir ("download_{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))

$curl = (Get-Command curl.exe -ErrorAction Stop).Source
& $curl --version | Select-Object -First 1 | Tee-Object -FilePath $Log

$ids = @(Get-Content $List | ForEach-Object { ($_ -split '\s+')[0] } | Where-Object { $_ -match '^\d{5}$' })
for ($pass = 1; $pass -le $Passes; $pass++) {
    $todo = @($ids | Where-Object {
        $f = Join-Path $OutDir "$_.tif"
        -not ((Test-Path $f) -and ($Expect -contains (Get-Item $f).Length))
    })
    Write-Host ("Pass {0}/{1}: {2} images in total; {3} to download" -f $pass, $Passes, $ids.Count, $todo.Count)
    if ($todo.Count -eq 0) { break }
    if ($pass -gt 1) { Add-Content -LiteralPath $Log -Value ("pass {0}: downloading again {1}" -f $pass, ($todo -join ' ')) -Encoding UTF8; Start-Sleep -Seconds 10 }
    for ($i = 0; $i -lt $todo.Count; $i += $BatchSize) {
        $batch = @($todo[$i..([Math]::Min($i + $BatchSize, $todo.Count) - 1)])
        $cfg = Join-Path $env:TEMP "alaska2_batch.cfg"
        # In a curl configuration file, "\" inside a double-quoted string is an escape character; use "/" instead (accepted by Windows).
        $lines = foreach ($id in $batch) { "url = `"$BaseUrl/$id.tif`""; "output = `"$((Join-Path $OutDir "$id.tif") -replace '\\', '/')`"" }
        Set-Content -Path $cfg -Value $lines -Encoding ascii
        $eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'   # curl error lines (stderr) must not stop the script
        & $curl --parallel --parallel-max $Parallel --fail --silent --show-error --retry 5 --retry-delay 3 --config $cfg 2>&1 |
            ForEach-Object { "$_" } | Tee-Object -FilePath $Log -Append | Out-Null
        $ErrorActionPreference = $eap
        Write-Host ("Processed {0}/{1}" -f [Math]::Min($i + $BatchSize, $todo.Count), $todo.Count)
    }
}

$bad = @($ids | Where-Object {
    $f = Join-Path $OutDir "$_.tif"
    -not ((Test-Path $f) -and ($Expect -contains (Get-Item $f).Length))
})
if ($bad.Count -gt 0) {
    Write-Host ("{0} images still missing or with a wrong size after {1} passes: {2}. Log: {3}" -f $bad.Count, $Passes, (($bad | Select-Object -First 30) -join ' '), $Log) -ForegroundColor Yellow
    Add-Content -LiteralPath $Log -Value ("still missing: " + ($bad -join ' ')) -Encoding UTF8
    exit 1
} else {
    Write-Host ("All {0} images present." -f $ids.Count) -ForegroundColor Green
}
