@echo off
echo Starting BERT head-tail model training with FP16 precision...
echo Using RTX 3070Ti with mixed precision

REM Activate the virtual environment
call .venv_new\Scripts\activate

REM Set HuggingFace cache
set HF_HOME=%USERPROFILE%\.cache\huggingface
echo HuggingFace cache set to: %HF_HOME%

REM Create output directories
mkdir output\checkpoints 2>nul

REM Train BERT head-tail model with FP16
echo.
echo Training BERT head-tail model with FP16 precision...
echo ===========================================================
python -m src.train --model bert_headtail --config configs/bert_headtail.yaml --fp16 --epochs 2

echo.
echo Training complete. Check output directory for saved model.
echo Model path: %CD%\output\checkpoints\bert_headtail_fold0.pth 