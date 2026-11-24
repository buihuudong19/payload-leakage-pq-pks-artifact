@echo off
rem Clean-environment check of a release (see clean_env_check.ps1). Arguments are passed on, e.g. -Tag v1.0.2
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0clean_env_check.ps1" %*
pause
