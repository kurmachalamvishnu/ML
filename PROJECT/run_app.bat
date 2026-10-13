@echo off
title Anti-Money Laundering (AML) Sentinel Web Application
color 0b
echo ===============================================================================
echo       ANTI-MONEY LAUNDERING (AML) DETECTION SYSTEM - WEB APPLICATION
echo ===============================================================================
echo Students: K. VISHNU (2520030199) ^& U. VIDHATH (2520030597)
echo Guide:    NIRMALAJYOTHI NARISETTY
echo Course:   Machine Learning (B.Tech 2026-27), KL University
echo ===============================================================================
echo.
echo [1/2] Activating Python Virtual Environment...
cd /d "%~dp0"
call .venv\Scripts\activate.bat

echo [2/2] Starting Flask AML Sentinel Server on http://127.0.0.1:5000 ...
start "" http://127.0.0.1:5000
python app.py

pause
