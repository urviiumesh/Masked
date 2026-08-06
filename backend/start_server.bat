@echo off
cd /d "%~dp0"
call face_env\Scripts\activate.bat
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
