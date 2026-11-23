<#
.SYNOPSIS
  Creates the environment .venv-confirm (Python 3.12 managed by uv) for the confirmatory CPU experiments: G4, E1, E2/E3, E4.

.DESCRIPTION
  As scripts\gpu\setup_gpu.ps1: the system Python and PATH are not touched; everything lives in .tools\ and .venv-confirm\
  of the project. Installs requirements.txt + sealwatch 2026.6 (SRM, FLD ensemble; pulls in the CPU build of PyTorch)
  + scikit-learn + matplotlib, then runs the test suite and checks that the source matches the frozen profile.
  Disk space: about 1.5 GB.

.EXAMPLE
  # From the repository root, in a normal PowerShell:
  powershell -ExecutionPolicy Bypass -File scripts\confirm\setup_confirm.ps1
#>
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root   = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Tools  = Join-Path $Root '.tools'
$UvDir  = Join-Path $Tools 'uv'
$Uv     = Join-Path $UvDir 'uv.exe'
$Venv   = Join-Path $Root '.venv-confirm'
$Py     = Join-Path $Venv 'Scripts\python.exe'
$LogDir = Join-Path $Root 'results\confirm_setup'
$UvVersion = '0.9.5'
$UvSha256  = '515dc53d7553f1357d0abc1f70acd921fbb9e30230b1d9a08737236daa6ee920'

$env:UV_PYTHON_INSTALL_DIR = Join-Path $Tools 'python'
# The uv cache uses a short path: inside a deep project directory, the paths of some PyTorch files in the cache
# (.tools\uv-cache\archive-v0\<id>\torch\include\...) exceed 260 characters. Set PQPKS_UV_CACHE to choose another location.
$env:UV_CACHE_DIR          = if ($env:PQPKS_UV_CACHE) { $env:PQPKS_UV_CACHE } else { Join-Path $env:LOCALAPPDATA 'pqpks-uv-cache' }
$env:UV_PYTHON_PREFERENCE  = 'only-managed'
$env:UV_NO_CONFIG          = '1'
$env:UV_LINK_MODE          = 'copy'
$env:PYTHONUTF8            = '1'

New-Item -ItemType Directory -Force -Path $Tools, $UvDir, $LogDir | Out-Null
$Log = Join-Path $LogDir ("setup_{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Start-Transcript -Path $Log | Out-Null
function Invoke-Checked([string]$What, [scriptblock]$Cmd) {
    Write-Host "==> $What" -ForegroundColor Cyan
    & $Cmd
    if ($LASTEXITCODE -ne 0) { throw "Failed ($LASTEXITCODE): $What" }
}
try {
    $drive = (Get-Item $Root).PSDrive
    $freeGB = [math]::Round($drive.Free / 1GB, 1)
    Write-Host "Drive $($drive.Name): $freeGB GB free (the whole confirmatory run needs about 250 GB, see docs\reproduction.md)"
    $longOk = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -ErrorAction SilentlyContinue).LongPathsEnabled -eq 1
    $deepest = (Join-Path $Venv 'Lib\site-packages\torch\include\ATen\native\transformers\cuda\mem_eff_attention\iterators\predicated_tile_access_iterator_residual_last.h').Length
    Write-Host "Deepest expected path in .venv-confirm: $deepest characters; Windows long paths: $longOk"
    if ($deepest -ge 260 -and -not $longOk) { throw "Project path too long ($deepest >= 260 characters): move the directory closer to the drive root" }
    if (-not (Test-Path $Uv)) {
        $zip = Join-Path $Tools "uv-$UvVersion.zip"
        $url = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
        Write-Host "==> Downloading uv $UvVersion" -ForegroundColor Cyan
        Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
        $h = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
        if ($h -ne $UvSha256) { Remove-Item $zip; throw "SHA-256 of uv does not match: $h" }
        Expand-Archive -Path $zip -DestinationPath $UvDir -Force
        Remove-Item $zip
    }
    Invoke-Checked 'Install Python 3.12 (local)' { & $Uv python install 3.12 --no-bin --no-registry }   # nothing written to ~\.local\bin or the registry
    if (-not (Test-Path $Py)) { Invoke-Checked 'Create .venv-confirm' { & $Uv venv $Venv --python 3.12 } }
    Invoke-Checked 'Install requirements.txt' { & $Uv pip install --python $Py -r (Join-Path $Root 'requirements.txt') }
    Invoke-Checked 'Install sealwatch, scikit-learn, matplotlib' { & $Uv pip install --python $Py 'sealwatch==2026.6' 'scikit-learn==1.9.1' 'matplotlib' }
    Invoke-Checked 'Install the project package (editable)' { & $Uv pip install --python $Py -e $Root }
    Invoke-Checked 'Record versions' { & $Uv pip freeze --python $Py | Out-File -Encoding utf8 (Join-Path $LogDir 'requirements-confirm.lock') }
    Push-Location $Root
    try {
        Invoke-Checked 'Check that the source matches the frozen profile' { & $Py scripts\freeze_profile.py --check }
        Invoke-Checked 'Run the test suite' { & $Py -m pytest -q -p no:cacheprovider }
    } finally { Pop-Location }
    Write-Host 'Done. Next step: docs\reproduction.md.' -ForegroundColor Green
}
finally {
    Stop-Transcript | Out-Null
    Write-Host "Log: $Log"
}
