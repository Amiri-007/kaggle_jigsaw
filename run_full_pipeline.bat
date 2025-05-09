@echo off
echo Starting RDS Project full training pipeline with FP16 precision...
echo Using RTX 3070Ti with mixed precision

REM Activate the virtual environment
call .venv_new\Scripts\activate

REM Set HuggingFace cache
set HF_HOME=%USERPROFILE%\.cache\huggingface
echo HuggingFace cache set to: %HF_HOME%

REM Create output directories
mkdir output\checkpoints 2>nul
mkdir output\preds 2>nul
mkdir results 2>nul
mkdir figs 2>nul

REM Train BERT head-tail model with FP16
echo.
echo Step 1: Training BERT head-tail model with FP16 precision...
echo ===========================================================
python -m src.train --model bert_headtail --config configs/bert_headtail.yaml --fp16 --epochs 2

REM Generate pseudo-labels
echo.
echo Step 2: Generating pseudo-labels from BERT model...
echo ===========================================================
python scripts/pseudo_label.py --base-model output/checkpoints/bert_headtail_fold0.pth --unlabeled-csv data/train.csv --out-csv output/pseudo_bert.csv

REM Train LSTM-Capsule model with FP16
echo.
echo Step 3: Training LSTM-Capsule model with FP16 precision...
echo ===========================================================
python -m src.train --model lstm_caps --config configs/lstm_caps.yaml --fp16 --epochs 6

REM Train GPT-2 head-tail model with FP16
echo.
echo Step 4: Training GPT-2 head-tail model with FP16 precision...
echo ===========================================================
python -m src.train --model gpt2_headtail --config configs/gpt2_headtail.yaml --fp16 --epochs 2

REM Blend models with Optuna
echo.
echo Step 5: Blending model predictions with Optuna...
echo ===========================================================
python -m src.blend_optuna --pred-dir output/preds --ground-truth data/valid.csv --n-trials 200 --out-csv output/preds/blend_ensemble.csv

REM Generate fairness metrics
echo.
echo Step 6: Generating fairness metrics...
echo ===========================================================
python scripts/write_metrics.py --predictions output/preds/blend_ensemble.csv --model-name blend_ensemble

REM Generate visualization figures
echo.
echo Step 7: Generating visualization figures...
echo ===========================================================
python notebooks/04_generate_figures.py

REM Run model explainers
echo.
echo Step 8: Running model explainers (SHAP & LIME)...
echo ===========================================================
python scripts/run_explainers.py --model-path output/checkpoints/bert_headtail_fold0.pth --n-samples 500

REM Show final summary
echo.
echo ===========================================================
echo FINAL METRICS SUMMARY:
echo ===========================================================
if exist results\summary.tsv (
    type results\summary.tsv
) else (
    echo No metrics summary found. Check for errors in previous steps.
)

echo.
echo RDS FAIRNESS AUDIT PIPELINE COMPLETED
echo.
echo Output locations:
echo - Model checkpoints: %CD%\output\checkpoints\
echo - Model predictions: %CD%\output\preds\
echo - Fairness metrics: %CD%\results\
echo - Visualization figures: %CD%\figs\
echo - Model explainers: %CD%\output\explainers\
echo. 