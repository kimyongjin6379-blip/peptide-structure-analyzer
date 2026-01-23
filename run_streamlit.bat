@echo off
echo Starting Peptide Structure Analyzer...
echo.
cd /d %~dp0
streamlit run streamlit_app/app.py
pause
