@echo off
rem Installs this fork's engine for CPUs without AVX2 for `strata launch` and the app (packaging/install_avx_engine.py).
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
set "PY=python"
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if exist "python\python.exe" set "PY=python\python.exe"
%PY% packaging\install_avx_engine.py %*
if errorlevel 1 echo. & echo The engine was not installed: see the message above.
pause
