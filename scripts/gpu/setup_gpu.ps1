<#
.SYNOPSIS
  Creates a separate GPU environment for SRNet: Python 3.12 + PyTorch with CUDA 12.6, entirely inside the project directory.

.DESCRIPTION
  Does not touch the system Python, does not change PATH, needs no administrator rights.
  Everything lives in the project directory and can be removed by deleting two directories:
    .tools\      uv 0.9.5 (SHA-256 checked), Python 3.12 managed by uv, uv cache
    .venv-gpu\   Python 3.12 virtual environment: torch (+cu126), numpy, pillow (see requirements-gpu.txt)
  Log: results\gpu_bench\setup_<timestamp>.log; installed versions: scripts\gpu\requirements-gpu.lock

  Disk space: about 2.5 GB downloaded, 5 GB after installation (project drive).

.EXAMPLE
  # From the repository root, in a normal PowerShell (no Administrator needed):
  powershell -ExecutionPolicy Bypass -File scripts\gpu\setup_gpu.ps1
  # Install and run the short benchmark right away (about 3 minutes):
  powershell -ExecutionPolicy Bypass -File scripts\gpu\setup_gpu.ps1 -RunBench -Quick
#>
param(
    [switch]$RunBench,
    [switch]$Quick
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'                 # the progress bar slows Invoke-WebRequest down considerably
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root     = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Tools    = Join-Path $Root '.tools'
$UvDir    = Join-Path $Tools 'uv'
$Uv       = Join-Path $UvDir 'uv.exe'
$Venv     = Join-Path $Root '.venv-gpu'
$Py       = Join-Path $Venv 'Scripts\python.exe'
$ReqFile  = Join-Path $PSScriptRoot 'requirements-gpu.txt'
$LockFile = Join-Path $PSScriptRoot 'requirements-gpu.lock'
$LogDir   = Join-Path $Root 'results\gpu_bench'

$UvVersion = '0.9.5'
$UvSha256  = '515dc53d7553f1357d0abc1f70acd921fbb9e30230b1d9a08737236daa6ee920'   # uv-x86_64-pc-windows-msvc.zip
$TorchIndex = 'https://download.pytorch.org/whl/cu126'

# uv: Python, cache and configuration stay in .tools; the system Python is not used
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Tools 'python'
# The uv cache uses a short path: inside a deep project directory, the paths of some PyTorch files in the cache
# exceed 260 characters. Set PQPKS_UV_CACHE to choose another location.
$env:UV_CACHE_DIR          = if ($env:PQPKS_UV_CACHE) { $env:PQPKS_UV_CACHE } else { Join-Path $env:LOCALAPPDATA 'pqpks-uv-cache' }
$env:UV_PYTHON_PREFERENCE  = 'only-managed'
$env:UV_NO_CONFIG          = '1'
$env:UV_LINK_MODE          = 'copy'

New-Item -ItemType Directory -Force -Path $Tools, $UvDir, $LogDir | Out-Null
$Log = Join-Path $LogDir ("setup_{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
Start-Transcript -Path $Log | Out-Null

function Invoke-Checked([string]$What, [scriptblock]$Cmd) {
    Write-Host "==> $What" -ForegroundColor Cyan
    & $Cmd
    if ($LASTEXITCODE -ne 0) { throw "Failed ($LASTEXITCODE): $What" }
}

try {
    Write-Host "Project directory: $Root"

    # 0. Check the GPU and the disk
    Invoke-Checked 'Check the GPU (nvidia-smi)' { nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader }
    $drive = (Get-Item $Root).PSDrive
    $freeGB = [math]::Round($drive.Free / 1GB, 1)
    Write-Host "Drive $($drive.Name): $freeGB GB free"
    if ($freeGB -lt 8) { throw "At least 8 GB of free space is needed on drive $($drive.Name)." }

    # 1. uv (official release from GitHub, SHA-256 checked before extraction)
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
    Invoke-Checked 'uv version' { & $Uv --version }

    # 2. Python 3.12 managed by uv (in .tools\python) and the virtual environment .venv-gpu
    Invoke-Checked 'Install Python 3.12 (local)' { & $Uv python install 3.12 --no-bin --no-registry }   # nothing written to ~\.local\bin or the registry
    if (-not (Test-Path $Py)) {
        Invoke-Checked 'Create .venv-gpu' { & $Uv venv $Venv --python 3.12 }
    }

    # 3. Packages: torch from the CUDA 12.6 index; the other packages from PyPI (read from requirements-gpu.txt)
    $reqs  = Get-Content $ReqFile | ForEach-Object { ($_ -split '#')[0].Trim() } | Where-Object { $_ }
    $torch = $reqs | Where-Object { $_ -match '^torch(==|>=|$)' }
    $rest  = $reqs | Where-Object { $_ -notmatch '^torch(==|>=|$)' }
    Invoke-Checked "Install $torch (+cu126)" { & $Uv pip install --python $Py $torch --index-url $TorchIndex }
    Invoke-Checked "Install $($rest -join ', ')" { & $Uv pip install --python $Py @rest }
    Invoke-Checked 'Ghi requirements-gpu.lock' { & $Uv pip freeze --python $Py | Out-File -Encoding utf8 $LockFile }

    # 4. Check that PyTorch sees the GPU and can run a matrix product on it
    $check = @'
import sys, torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.version.cuda, "| cudnn", torch.backends.cudnn.version())
assert torch.cuda.is_available(), "torch.cuda.is_available() = False"
p = torch.cuda.get_device_properties(0)
print("gpu", p.name, "| VRAM %.1f GB" % (p.total_memory / 2**30), "| SM", p.multi_processor_count, "| compute capability %d.%d" % (p.major, p.minor))
a = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize(); print("matmul ok:", float((a @ a).abs().mean()) > 0)
'@
    Invoke-Checked 'Check PyTorch + CUDA' { $check | & $Py - }

    Write-Host ''
    Write-Host 'Done. Next step (close other GPU applications, connect the charger, Best performance power mode):' -ForegroundColor Green
    Write-Host "  $Py scripts\gpu\bench_srnet.py --quick   # about 3 minutes"
    Write-Host "  $Py scripts\gpu\bench_srnet.py           # full benchmark, about 15 minutes"

    if ($RunBench) {
        $benchArgs = @((Join-Path $PSScriptRoot 'bench_srnet.py'))
        if ($Quick) { $benchArgs += '--quick' }
        Push-Location $Root
        try { Invoke-Checked 'SRNet benchmark' { & $Py @benchArgs } } finally { Pop-Location }
    }
}
finally {
    Stop-Transcript | Out-Null
    Write-Host "Log: $Log"
}
