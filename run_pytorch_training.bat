@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "LOG=%~dp0training_run.log"
echo [%date% %time%] Training started > "%LOG%"

echo ==========================================
echo CIFAR-100 PyTorch Training
echo ==========================================
echo Log file: %LOG%
echo.

set "PY=python"
where python >nul 2>&1
if errorlevel 1 (
    set "PY=py -3"
    where py >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python not found. Install Python and add it to PATH.
        echo [ERROR] Python not found. >> "%LOG%"
        pause
        exit /b 1
    )
)

echo Checking PyTorch...
%PY% -c "import torch; print('PyTorch OK, CUDA:', torch.cuda.is_available())" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [ERROR] PyTorch not installed or broken.
    echo Install with:
    echo   %PY% -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo See log: %LOG%
    pause
    exit /b 1
)
%PY% -c "import torch; print('PyTorch OK, CUDA:', torch.cuda.is_available())"

echo.
echo [1/2] Training Coarse (20 classes), 20 epochs ...
%PY% cifar100_pytorch/train.py --label coarse --model resnet --epochs 20 --batch-size 64 >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [ERROR] Coarse training failed. See log: %LOG%
    pause
    exit /b 1
)

echo.
echo [2/2] Training Fine (100 classes), 20 epochs ...
%PY% cifar100_pytorch/train.py --label fine --model resnet --epochs 20 --batch-size 64 >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [ERROR] Fine training failed. See log: %LOG%
    pause
    exit /b 1
)

echo.
echo Done. Models saved in: %~dp0models
echo Error samples saved in: %~dp0outputs
echo Full log: %LOG%
echo [%date% %time%] Training finished OK >> "%LOG%"
pause