@echo off
cd /d "%~dp0"
call face_env\Scripts\activate.bat

echo Installing / switching to NVIDIA CUDA onnxruntime-gpu...
pip uninstall -y onnxruntime onnxruntime-directml 2>nul
pip install --upgrade onnxruntime-gpu

echo.
echo Verifying providers...
python -c "from api.services.runtime_device import runtime_info; import json; print(json.dumps(runtime_info(), indent=2))"

echo.
echo Done. Start the server with start_server.bat
pause
