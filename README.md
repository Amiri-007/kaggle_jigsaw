# RDS Project: Deep Fairness for Toxicity Classification

This project provides a modern implementation for toxicity classification with deep learning models, focusing on bias reduction, fairness evaluation across demographic groups, and model explainability using SHAP. It builds upon the [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) dataset.

## Project Structure

The codebase has been reorganized for better clarity and maintainability. See [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md) for the complete organization.

```
.
├── src/                     # Core source code
│   ├── train.py             # Training pipeline implementation
│   ├── predict.py           # Prediction script
│   ├── blend_optuna.py      # Model blending optimization
│   ├── data/                # Data loading and processing
│   ├── models/              # Model implementations
│   │   ├── lstm_caps.py     # LSTM-Capsule model
│   │   ├── bert_headtail.py # BERT head-tail model
│   │   └── gpt2_headtail.py # GPT-2 head-tail model
├── configs/                 # Configuration files for different models
├── data/                    # Dataset files (download from Kaggle)
├── pipelines/               # Pipeline execution scripts
│   ├── run_turbo.ps1        # Main turbo model pipeline
│   └── run_turbo_large.ps1  # Large dataset turbo pipeline
├── explainers/              # Model explainability tools
│   ├── run_simplified_explainer.py  # Token attribution analysis
│   ├── run_turbo_shap.py    # SHAP analysis for turbo model
│   └── run_turbo_shap_simple.py     # Simplified token importance analysis
├── predictions/             # Custom prediction scripts
│   ├── run_custom_predict.py        # Prediction script
│   └── large_predict.py             # Prediction for larger datasets
├── output/                  # Model outputs (checkpoints, predictions)
└── archive/                 # Archived scripts and tools
    ├── analysis/            # Analysis tools
    ├── fairness/            # Fairness metrics and tools
    ├── scripts/             # Utility scripts
    └── misc/                # Miscellaneous utilities
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Amiri-007/kaggle_jigsaw.git
cd kaggle_jigsaw

# Create virtual environment and activate it
python -m venv .venv_new && .\.venv_new\Scripts\activate.ps1  # For Windows
# OR
python -m venv .venv && source .venv/bin/activate  # For Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Quick Start: Turbo Mode

The project implements a "turbo mode" for rapid development and testing, which uses smaller models and data samples:

```bash
# Run the full turbo pipeline
.\pipelines\run_turbo.ps1  # Windows PowerShell
```

### Turbo Pipeline Steps

The turbo mode runs the complete pipeline with the following steps:

1. **DistilBERT Training**: Trains a lightweight DistilBERT model on a 5% sample of the data
   ```
   python -m src.train --model bert_headtail --config configs/bert_headtail_turbo.yaml --fp16 --turbo
   ```

2. **LSTM-Capsule Training**: Trains an LSTM-Capsule model on the same data sample
   ```
   python -m src.train --model lstm_caps --config configs/lstm_caps_turbo.yaml --turbo
   ```
   
3. **GPT-2 Training**: Trains a GPT-2 model in the same configuration
   ```
   python -m src.train --model gpt2_headtail --config configs/gpt2_headtail_turbo.yaml --fp16 --turbo
   ```

4. **Model Blending**: Uses Optuna to find optimal model weights
   ```
   python -m src.blend_optuna --pred-dir output/preds --ground-truth data/valid.csv --n-trials 10
   ```

5. **Fairness Metrics**: Calculates comprehensive fairness metrics
   ```
   python scripts/write_metrics.py --pred output/preds/blend_turbo.csv --model-name blend_turbo
   ```

6. **Visualization**: Generates figures for model performance and fairness
   ```
   jupyter nbconvert --execute notebooks/tmp_figs.ipynb --to html
   ```

### Turbo Configuration

The turbo mode uses specialized configurations:

- **bert_headtail_turbo.yaml**: Uses `distilbert-base-uncased`, 5% data sample, FP16 precision
- **lstm_caps_turbo.yaml**: Uses smaller embedding dimensions and fewer epochs
- **gpt2_headtail_turbo.yaml**: Uses `distilgpt2` with reduced sequence length

## Model Explainability

We've implemented multiple approaches for model interpretability:

1. **Simplified Token Importance Analysis**: Fast and reliable token importance visualization for the turbo model
   ```bash
   # Run with PowerShell script
   .\pipelines\run_turbo_shap_simple.ps1
   
   # Or directly with Python
   python explainers/run_turbo_shap_simple.py --ckpt output/checkpoints/distilbert_headtail_fold0.pth
   ```
   
   This approach uses a token occlusion technique to identify the most important tokens and generates:
   - Token importance bar charts showing the effect of each token on prediction
   - Color-coded text visualizations
   - Detailed token importance files

2. **Turbo Model SHAP Analysis**: More advanced SHAP analysis (may be less stable)
   ```bash
   # Run with PowerShell script
   .\pipelines\run_turbo_shap.ps1
   
   # Or directly with Python
   python explainers/run_turbo_shap.py --ckpt output/checkpoints/distilbert_headtail_fold0.pth
   ```

3. **Other SHAP Approaches**: Additional implementation details
   ```bash
   python explainers/run_simplified_explainer.py --ckpt output/checkpoints/distilbert_headtail_fold0.pth
   python explainers/generate_mock_shap.py --model_path output/checkpoints/distilbert_headtail_fold0.pth
   ```

These explainers generate visualizations including:
- Token importance charts
- Waterfall plots showing token contributions
- Heatmaps of token influence

## Large Dataset Pipeline

For running on larger datasets:

```bash
# Run the large dataset pipeline (8% vs 5% in original turbo mode)
.\pipelines\run_turbo_large.ps1
```

For custom predictions on large datasets:

```bash
# Run custom predictions using existing model checkpoint
python predictions/run_custom_predict.py --checkpoint output/checkpoints/distilbert_headtail_fold0.pth
```

## Features

- **State-of-the-art models**: LSTM-Capsule with EMA, BERT and GPT-2 with head-tail architecture
- **Advanced training techniques**: Negative downsampling, weighted training, pseudo-labeling
- **Comprehensive fairness evaluation**: Subgroup AUC, BPSN/BNSP metrics, threshold analysis
- **Model explainability**: SHAP analysis and token attributions
- **Optimized pipelines**: Fast turbo mode, large dataset handling, mixed-precision training

## License

MIT License

## References

- [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
- [Perspective API](https://perspectiveapi.com/)
- [Kaggle 3rd Place Solution](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/discussion/97471)
- [SHAP: SHapley Additive exPlanations](https://github.com/slundberg/shap)
