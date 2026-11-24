@echo off
rem Steps registered after review (RV1-RV10), see run_review_steps.ps1. Example: run_review_steps.cmd -Step RV1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_review_steps.ps1" %*
pause
