# RDS Project: Fairness Audit of Jigsaw Toxicity Classifier

**Project**: Audit of unintended bias in toxic comment classification using the Jigsaw Civil Comments dataset. This repository contains the full implementation of a toxicity classification model and a comprehensive fairness evaluation, as described in the report *"Fairness Audit: Jigsaw Unintended Bias in Toxicity Classification."* We train a state-of-the-art text toxicity detector and then assess its performance across demographic subgroups for bias, using both traditional fairness metrics and model explainability (SHAP values).

## Overview and Purpose

Online toxicity detection models can inadvertently behave unfairly toward certain protected groups (e.g., flagging non-toxic comments about specific races or religions as toxic). This project builds a **toxic comment classifier** and then performs a rigorous **fairness audit**. Key aspects include:

* **Model Implementation:** A BERT-based classification model (with comparisons to LSTM and GPT-2 variants) trained on the Civil Comments dataset.
* **Bias Metrics:** Evaluation of model bias using subgroup AUC metrics (from Jigsaw competition) and classic fairness measures like demographic parity and error rate parity.
* **Explainability:** Use of SHAP (Shapley Additive Explanations) to interpret model predictions and a custom "**SHarP**" score (SHAP-based fairness metric) to quantify how differently the model treats identity-related content.

The goal is to identify bias in the model's behavior and suggest ways to mitigate such unintended biases. This repository is structured to allow anyone to **reproduce our results and figures** and inspect the code for each component of the analysis.

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
├── fairness_analysis/       # Fairness evaluation and bias auditing tools
│   ├── run_fairness_analysis.py      # Complete fairness analysis pipeline
│   ├── run_sharp_analysis.py         # SHarP individual fairness analysis
│   ├── metrics_v2.py                 # Fairness metrics implementation
│   ├── audit_fairness_v2.py          # Fairness metrics calculation
│   └── shap_report.py                # SHAP fairness reporting
├── explainability/          # Model explainability tools
│   ├── run_simplified_explainer.py   # Token attribution analysis
│   ├── run_turbo_shap.py             # SHAP value generation and analysis
│   ├── run_turbo_shap_simple.py      # Simplified token importance analysis
│   └── explainers_distilbert.py      # DistilBERT-specific explainers
├── predictions/             # Custom prediction scripts
│   ├── run_custom_predict.py        # Prediction script
│   └── large_predict.py             # Prediction for larger datasets
├── output/                  # Output files (checkpoints, predictions, metrics)
│   ├── models/              # Model checkpoints
│   ├── predictions/         # Model predictions
│   ├── metrics/             # Evaluation metrics
│   ├── figures/             # Generated visualizations
│   └── explainability/      # SHAP analysis outputs
├── results/                 # Results of experiments and analysis
└── archive/                 # Archived scripts and tools
    ├── analysis/            # Analysis tools
    ├── fairness/            # Legacy fairness metrics and tools
    ├── scripts/             # Utility scripts
    └── misc/                # Miscellaneous utilities
```
## To run the main notebook download the following file and the kaggle all_data file (see description below)
- [simplest_preds.csv](https://drive.google.com/file/d/1f-4iuqBLSEl3gSZrPsqv1m3jFKT8DEzx/view?usp=sharing)

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

**Note:** The pipeline pulls the dataset via the Kaggle CLI.  
If you have never used it, run `pip install kaggle && kaggle config set -n path -v ~/.kaggle`
and place your `kaggle.json` API token in that folder.

## Quick Start

### 1. Grab the Civil-Comments dataset (one-time)

```bash
# generates ~/.kaggle/kaggle.json the first time you run it
make data
```

> **Heads-up ⚠️**
> You need a free Kaggle account.
> • Go to **My Account → Create New API Token** to download `kaggle.json`.
> • When the script prompts, paste the file's **whole contents** and press <kbd>Ctrl-D</kbd> (Linux/macOS) or <kbd>Ctrl-Z</kbd> then <kbd>Enter</kbd> (Windows Powershell).

### 2. Reproducible Pipeline

```bash
# 1. clone & create virtual-env
git clone https://github.com/Amiri-007/kaggle_jigsaw.git
cd kaggle_jigsaw
make ENV=.venv           # creates venv & installs pinned deps

# 2. place Kaggle API token
mkdir -p ~/.kaggle && cp path/to/kaggle.json ~/.kaggle && chmod 600 ~/.kaggle/kaggle.json

# 3. full pipeline: data → train → fairness + SHAP
make rc-all

# artifacts:
#   figs/      ← publication-quality plots used in report
#   results/   ← CSVs with AUROCs, SHAPr divergence, etc.
```

| Path | Purpose |
|------|---------|
| `src/train.py` | Train DistilBERT classifier (turbo config) |
| `scripts/merge_preds_with_labels.py` | Join model predictions with identity labels for audit |
| `fairness_analysis/run_sharp_analysis.py` | Compute SHAP values & SHAPr divergence plots |
| `fairness_analysis/metrics.py` | Core fairness metrics (Subgroup/BPSN/BNSP AUC etc.) |
| `Makefile` | One-command pipeline (`make rc-all`) |

### 3. Train / reproduce results

```bash
make full-run FP16=1         # see configs/*.yaml for knobs
```

## Turbo Mode: Quick Testing

The project implements a "turbo mode" for rapid development and testing, which uses smaller models and data samples:

```bash
# Run the full turbo pipeline
# Linux/macOS:
make turbo-run

# Windows (PowerShell):
.\pipelines\run_turbo.ps1
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
   python fairness_analysis/audit_fairness_v2.py --preds output/preds/blend_turbo.csv --val data/valid.csv
   ```

6. **Visualization**: Generates figures for model performance and fairness
   ```
   jupyter nbconvert --execute notebooks/generate_figures.ipynb --to html
   ```

## Fairness Analysis

The fairness analysis tools provide comprehensive fairness evaluation and auditing:

1. **Complete Fairness Analysis Pipeline**: Run the entire fairness analysis suite
   ```bash
   python fairness_analysis/run_fairness_analysis.py --model your_model_name
   ```
   
   This will:
   - Count and visualize demographic distribution
   - Run fairness auditing with metrics calculation
   - Analyze intersectional fairness
   - Check compliance with fairness requirements
   - Launch the interactive fairness dashboard

2. **Fairness Metrics**: Calculate specific fairness metrics
   ```bash
   python fairness_analysis/audit_fairness_v2.py --preds results/your_model_preds.csv --val data/your_validation.csv --thr 0.6 --majority white
   ```
   
   This calculates:
   - Selection rates by demographic group
   - False positive and negative rates (FPR/FNR)
   - Demographic parity difference and ratios
   - Disparities relative to majority group
   - 80% rule compliance

3. **SHarP Individual Fairness Analysis**: Analyze how attribution patterns differ across demographic groups
   ```bash
   python fairness_analysis/run_sharp_analysis.py --sample 2000
   ```
   
   The SHarP analysis:
   - Computes SHAP values for a sample of validation examples
   - Calculates mean attribution vectors for each demographic subgroup
   - Computes divergence scores showing which groups have the most different attribution patterns
   - Visualizes the results and generates detailed reports

4. **Identity Prevalence Comparison**: Compare toxicity prevalence across different identity groups
   ```bash
   python fairness_analysis/compare_identity_prevalence.py
   ```
   
   The identity prevalence comparison:
   - Computes toxicity prevalence for specific identity subgroups
   - Calculates risk ratios relative to background prevalence
   - Visualizes comparative prevalence with confidence intervals
   - Generates metrics and summary reports for inclusion in analysis

## Model Explainability

We've implemented multiple approaches for model interpretability:

1. **Simplified Token Importance Analysis**: Fast and reliable token importance visualization for the turbo model
   ```bash
   # Run with PowerShell script
   .\pipelines\run_turbo_shap_simple.ps1
   
   # Or directly with Python
   python explainability/run_turbo_shap_simple.py --ckpt output/checkpoints/distilbert_headtail_fold0.pth
   ```
   
   This approach uses a token occlusion technique to identify the most important tokens and generates:
   - Token importance bar charts showing the effect of each token on prediction
   - Color-coded text visualizations
   - Detailed token importance files

2. **Turbo Model SHAP Analysis**: More advanced SHAP analysis
   ```bash
   # Run with PowerShell script
   .\pipelines\run_turbo_shap.ps1
   
   # Or directly with Python
   python explainability/run_turbo_shap.py --ckpt output/checkpoints/distilbert_headtail_fold0.pth
   ```

3. **SHarP Analysis with Cross-Platform Compatibility**: Enhanced SHAP-based fairness analysis with robust model checkpoint loading
   ```bash
   # Run with standard parameters
   python fairness_analysis/run_sharp_analysis.py --sample 2000
   
   # Run with custom model path and sample size
   python fairness_analysis/run_sharp_analysis.py --model-path output/checkpoints/your_model.pth --sample-size 1000
   ```
   
   The SHarP analysis script features enhanced model checkpoint loading that:
   - Supports multiple checkpoint formats (models saved from different training pipelines)
   - Handles distributed training checkpoints (with 'module.' prefix)
   - Provides detailed error diagnostics and fallback loading strategies
   - Works across both Windows and Linux/macOS environments

These explainers generate visualizations including:
- Token importance charts
- Waterfall plots showing token contributions
- Heatmaps of token influence
- SHarP divergence bar charts for demographic groups

## Fairness Metrics and Considerations

The fairness analysis tools calculate various metrics across demographic groups:

1. **Group Fairness Metrics**
   - **Selection Rates**: Percentage of positive predictions for each group
   - **Demographic Parity Difference**: Absolute difference in selection rates
   - **Demographic Parity Ratio**: Ratio of selection rates (for 80% rule compliance)

2. **Error-based Fairness Metrics**
   - **False Positive Rate (FPR)**: Rate of false positives for each group
   - **False Negative Rate (FNR)**: Rate of false negatives for each group
   - **Error Rate Disparities**: Differences in error rates across groups

3. **Kaggle-specific Bias Metrics**
   - **Subgroup AUC**: AUC score for specific identity subgroups
   - **BPSN AUC**: Background Positive, Subgroup Negative AUC
   - **BNSP AUC**: Background Negative, Subgroup Positive AUC
   - **Final Bias Score**: Combined bias metric used in the competition

4. **Individual Fairness**
   - **SHarP Divergence**: Measures differences in feature attribution patterns

When interpreting fairness metrics, consider:
- **Intersectionality**: Different aspects of identity often intersect
- **Context**: The importance of metrics depends on the specific context
- **Tradeoffs**: Different fairness metrics may be in tension with each other
- **Sample Size**: Groups with few samples may have unreliable metrics

## Large Dataset Pipeline

For running on larger datasets:

```bash
# Run the large dataset pipeline (8% vs 5% in original turbo mode)
# Linux/macOS:
make full-run

# Windows (PowerShell):
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
- **Individual fairness analysis**: SHarP divergence metrics across demographic groups
- **Optimized pipelines**: Fast turbo mode, large dataset handling, mixed-precision training

## License

MIT License

## References

- [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
- [Kaggle 3rd Place Solution](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/discussion/97471)
- [Perspective API](https://perspectiveapi.com/)
- [Fairness Definitions Explained](https://fairware.cs.umass.edu/papers/Verma.pdf)
- [Fairness and Machine Learning](https://fairmlbook.org/)
- [SHAP: SHapley Additive exPlanations](https://github.com/slundberg/shap)

## SHarP Analysis: Cross-Platform Compatibility

The SHarP (SHAP-based Fairness) analysis tools have been enhanced to work seamlessly across different platforms (Windows/Linux/macOS) with various model checkpoint formats. These improvements provide:

1. **Robust Model Loading**: The model loading function can now handle various checkpoint formats and state dictionary structures.
   - Works with both DistilBERT and BERT-based models
   - Handles checkpoints from distributed training with module prefix
   - Automatically applies fallback loading with strict=False when needed

2. **Test Scripts**: Verify model loading and SHarP analysis functionality:
   ```powershell
   # Windows
   .\test_sharp_loading.ps1
   ```
   ```bash
   # Linux/macOS
   python scripts/test_sharp_loading.py --model-path output/checkpoints/distilbert_headtail_fold0.pth
   ```

3. **Simplified Testing Mode**: When running with small samples (--sample-size < 10), a simplified simulation mode is used to generate representative visualizations even when data is limited.

4. **Better Error Handling**: All functions have improved error recovery and diagnostic messages for troubleshooting model compatibility issues.

The improved SHarP analysis script can be run directly with:

```powershell
# Windows
python fairness_analysis/run_sharp_analysis.py --sample-size 500
```

```bash
# Linux/macOS
python fairness_analysis/run_sharp_analysis.py --sample-size 500

This will generate model explanation visualizations and fairness divergence metrics that measure how differently the model "reasons" for different demographic groups.
``` 
