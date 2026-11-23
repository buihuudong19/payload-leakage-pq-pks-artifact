@echo off
rem Runs run_all.ps1 (parallel lanes + monitor window). Double-click to start or resume; arguments: -Reset, -DryRun, -NoMonitor
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_all.ps1" %*
pause
