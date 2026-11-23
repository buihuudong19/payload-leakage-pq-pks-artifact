#!/usr/bin/env bash
# Runs the whole confirmatory step of PQ-PKS (parallel lanes + monitor window).
#
# The environments .venv-confirm, .venv-gpu and CUDA of the measurement machine are Windows environments, so the orchestration is
# written in PowerShell (run_all.ps1). This file only calls run_all.ps1, so that it can be started from Git Bash or WSL:
#     ./run_all.sh --dry-run     # test the orchestration without running any command
#     ./run_all.sh --reset       # full run from the beginning
#     ./run_all.sh               # resume after an interruption
# From PowerShell call it directly: powershell -ExecutionPolicy Bypass -File .\run_all.ps1 [-Reset] [-DryRun] [-NoMonitor]
set -euo pipefail
cd "$(dirname "$0")"

args=()
for a in "$@"; do
  case "$a" in
    --reset) args+=("-Reset") ;;
    --dry-run) args+=("-DryRun") ;;
    --no-monitor) args+=("-NoMonitor") ;;
    *) echo "unknown argument: $a (use --reset, --dry-run, --no-monitor)" >&2; exit 2 ;;
  esac
done

if command -v powershell.exe >/dev/null 2>&1; then
  ps1="run_all.ps1"
  if command -v wslpath >/dev/null 2>&1; then ps1="$(wslpath -w "$PWD/run_all.ps1")"; fi
  exec powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$ps1" ${args[@]+"${args[@]}"}
fi
echo "powershell.exe not found: run_all needs Windows (the Python and CUDA environments of the project are Windows environments)." >&2
exit 1
