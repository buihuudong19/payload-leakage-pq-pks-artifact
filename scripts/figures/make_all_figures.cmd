@echo off
rem Regenerate every figure of the PQ-PKS paper (Windows). See make_all_figures.py.
setlocal
set "ROOT=%~dp0..\.."
set "PY=%ROOT%\.venv-confirm\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" "%~dp0make_all_figures.py" %*
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" (echo Done. Figures are in results\figures and results\figures_confirm.) else (echo An error occurred, see the messages above.)
pause
exit /b %RC%
