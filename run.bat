@echo off
echo ========================================
echo  Real-Time Vehicle Inspection Pipeline
echo ========================================

set FLASK_APP=app/main.py
set UPLOAD_DIR=app/static/uploads
set OUTPUT_DIR=app/static/outputs
set PORT=8000

REM Defect Detection Configuration (FasterRCNN)
set MODEL_TYPE=fasterrcnn
set MODEL_PATH=fasterrcnn_model.pth
set CONFIDENCE_THRESHOLD=0.5

REM SAM2 Surface Defect Detection Configuration
set SAM2_MODEL=facebook/sam2-hiera-tiny

echo Configuration:
echo - Model Type: %MODEL_TYPE% (FasterRCNN)
echo - Model Path: %MODEL_PATH%
echo - Confidence Threshold: %CONFIDENCE_THRESHOLD%
echo - SAM2 Model: %SAM2_MODEL%
echo - Port: %PORT%
echo.
echo Detection Modes Available: 
echo   * FasterRCNN Only: Fast structural defect detection (Dents, Scratches)
echo   * FasterRCNN + SAM2: Complete inspection with surface analysis
echo     - Paint Defects, Contamination, Corrosion, Water Spots
echo.
echo Features:
echo   * Real-time camera feed
echo   * Live capture and analysis
echo   * Upload mode (legacy support)
echo.
echo Starting Real-Time Inspection Server...
echo Access the application at: http://localhost:%PORT%
echo.
echo Camera Setup Test (optional):
echo   python test_camera_setup.py
echo.

set PYTHONPATH=.
python app/main.py


@REM cd InspecAI_v2
@REM .\venv\Scripts\Activate.ps1
@REM .\run.bat