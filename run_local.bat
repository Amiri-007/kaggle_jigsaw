@echo off
echo ===================================================
echo Jigsaw Unintended Bias Audit - GPU Runner
echo ===================================================
echo.

REM Parse command-line arguments
set SKIP_BERT=false
set SECURE_KAGGLE=false
set NROWS=50000

:arg_loop
if "%1"=="" goto arg_done
if /i "%1"=="--skip-bert" set SKIP_BERT=true
if /i "%1"=="--secure-kaggle" set SECURE_KAGGLE=true
if /i "%1"=="--nrows" set NROWS=%2 & shift
shift
goto arg_loop
:arg_done

REM Check if virtual environment exists
if not exist venv (
    echo Virtual environment not found!
    echo Please run setup.bat first to create the environment.
    pause
    exit /b
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install required packages with explicit versions
echo Installing required packages...
pip install -r requirements.txt
pip install seaborn==0.12.2

REM Setup Kaggle API credentials if requested
if "%SECURE_KAGGLE%"=="true" (
    echo Setting up secure Kaggle credentials...
    python secure_kaggle.py --setup
)

REM Force CUDA environment variables to ensure GPU detection
set CUDA_VISIBLE_DEVICES=0
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

REM Verify CUDA availability
echo.
echo Verifying CUDA availability...
python -c "import torch; torch.cuda.set_device(0); gpu_available = torch.cuda.is_available(); device_name = torch.cuda.get_device_name(0) if gpu_available else 'None'; print(f'CUDA Available: {gpu_available}'); print(f'GPU Device: {device_name}')"

REM Check if data directory exists
if not exist data (
    echo Data directory not found!
    echo Running environment setup...
    echo.
    echo NOTE: You will need to provide your Kaggle API key.
    echo If you don't have one, visit https://www.kaggle.com/settings
    echo and create a new API token.
    echo.
    python setup_environment.py --data_dir ./data
)

REM Set options based on configuration
set OPTIONS=--data_dir ./data --nrows %NROWS% --batch_size 16
if "%SKIP_BERT%"=="true" set OPTIONS=%OPTIONS% --skip_bert

REM Run the analysis with the selected options
echo.
echo Running Jigsaw Unintended Bias Analysis on GPU...
echo Using batch_size=16 and nrows=%NROWS%
if "%SKIP_BERT%"=="true" echo Skipping BERT inference (TF-IDF model only)
echo.
python run_analysis.py %OPTIONS%

echo.
echo ===================================================
echo Analysis complete!
echo Results saved to output directory.
echo For more options, run: python run_analysis.py --help
echo ===================================================
echo.

pause 