@echo off
cd /d "%~dp0"
call face_env\Scripts\activate.bat

echo Installing / switching to DirectML GPU runtime (Intel/AMD/NVIDIA on Windows)...
pip uninstall -y onnxruntime onnxruntime-gpu 2>nul
pip install --upgrade onnxruntime-directml

echo.
echo Verifying providers...
python -c "from api.services.runtime_device import runtime_info; import json; print(json.dumps(runtime_info(), indent=2))"

echo.
echo Done. Start the server with start_server.bat
pause
