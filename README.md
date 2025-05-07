# Jigsaw Unintended Bias Audit

This project analyzes the [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) dataset to evaluate potential biases in machine learning models across demographic groups.

## Optimized for RTX 3070Ti GPU & Windows

This project is specifically optimized to run on:
- NVIDIA RTX 3070Ti GPU (laptop or desktop)
- Windows 10/11
- Python 3.10+

## Features

- Analyzes bias in toxicity classification models across 13 demographic groups
- Provides comprehensive metrics: Subgroup AUC, BPSN, BNSP, and Power Difference
- Calculates optimal classification thresholds that minimize bias
- Includes detailed visualizations to help understand model biases
- Supports both TF-IDF+LogReg and BERT models
- GPU-optimized for fast computation

## Secure Setup

This project uses Kaggle API credentials but handles them securely:

1. **Never commits API keys to GitHub**
2. Supports multiple secure methods for providing credentials:
   - Environment variables (`KAGGLE_USERNAME` and `KAGGLE_KEY`)
   - Interactive setup through a secure helper script
   - Local configuration file (automatically gitignored)

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Amiri-007/kaggle_jigsaw.git
cd kaggle_jigsaw
```

### 2. Set up with the automated script

```bash
setup.bat
```

The setup script will:
- Create a Python virtual environment
- Install the required dependencies
- Prompt for Kaggle API credentials (if not found)
- Download the dataset

### 3. Run the analysis

```bash
# Basic usage (TF-IDF + BERT models with GPU)
run_local.bat

# Skip BERT inference (much faster, only TF-IDF model)
run_local.bat --skip-bert

# Use more/less data
run_local.bat --nrows 100000

# Set up secure Kaggle credentials if not already configured
run_local.bat --secure-kaggle
```

## Fairness Dashboard (New for 2025)

We've added a comprehensive fairness dashboard to visualize and analyze bias metrics across demographic subgroups:

### Key Features

- **Bias Metrics Visualization**: Interactive heatmaps showing Subgroup AUC, BPSN AUC, and BNSP AUC across all identity groups
- **Threshold Playground**: Interactive tool to find optimal classification thresholds that minimize bias
- **Power Mean Analysis**: Aggregation of bias metrics using generalized power means with configurable parameters
- **Model Comparison**: Compare bias metrics across different model architectures
- **Exportable Reports**: Save results as CSV files for further analysis

### Running the Fairness Dashboard

```bash
# Install Streamlit (if not already installed)
pip install -r app/requirements.txt

# Run the dashboard
cd app
streamlit run app.py
```

### Bias Evaluation Notebook

A Jupyter notebook is provided for detailed bias evaluation:

```bash
# Run the notebook with a specific model
cd notebooks
papermill 03_bias_evaluation.ipynb output.ipynb -p model_name "tfidf_logreg"
```

### Bias Metrics API

The enhanced metrics module (`src/metrics_v2.py`) provides a vectorized implementation of bias metrics that can be integrated into any Python project:

```python
import numpy as np
from src.metrics_v2 import compute_all_metrics

# Example usage
results = compute_all_metrics(
    y_true=y_true,               # Ground truth labels
    y_pred=y_pred,               # Model predictions
    subgroup_masks=subgroup_masks,  # Dictionary mapping subgroup names to boolean masks
    power=-5,                    # Power parameter for generalized mean
    weight_overall=0.25          # Weight for overall AUC in final score
)

# Access results
overall_auc = results["overall"]["auc"]
final_score = results["overall"]["final_score"]
subgroup_metrics = results["subgroup_metrics"]  # List of per-subgroup metrics
```

### Fairness Metrics Workflow

1. Train models using standard workflows
2. Generate predictions on test data
3. Run `03_bias_evaluation.ipynb` to calculate bias metrics
4. Launch the Streamlit dashboard to explore results interactively

## Project Structure

```
.
├── app/                      # Streamlit dashboard application
│   ├── app.py                # Main dashboard
│   └── requirements.txt      # Dashboard dependencies
├── artifacts/                # Saved model artifacts (vectorizers, models)
├── data/                     # Dataset files (downloaded via setup)
├── figs/                     # Output figures and visualizations
├── logs/                     # Training logs and validation results
├── models/                   # Trained model checkpoints
├── notebooks/                # Jupyter notebooks
│   └── 03_bias_evaluation.ipynb # Fairness metrics notebook 
├── results/                  # Evaluation results (metrics, reports)
├── src/                      # Source code
│   ├── metrics_v2.py         # Enhanced bias metrics module
│   └── vis_utils.py          # Visualization utilities
├── tests/                    # Unit tests
│   └── test_metrics_v2.py    # Tests for bias metrics
├── bias_metrics.py           # Original metrics for bias evaluation
├── plot_utils.py             # Original visualization utilities
├── requirements.txt          # Python dependencies
├── run_analysis.py           # The main analysis script
├── run_local.bat             # Windows runner script
├── secure_kaggle.py          # Secure handling of Kaggle credentials
├── setup_environment.py      # Environment setup script
├── setup.bat                 # Windows setup script
└── Jigsaw_Unintended_Bias_Audit.ipynb # Main notebook (optional)
```

## Secure Credential Handling

This project offers multiple options for handling Kaggle API credentials securely:

### Option 1: Environment Variables

Set `KAGGLE_USERNAME` and `KAGGLE_KEY` as environment variables:

```bash
# Windows (PowerShell)
$env:KAGGLE_USERNAME = "yourusername"
$env:KAGGLE_KEY = "yourapikey"

# Windows (CMD)
set KAGGLE_USERNAME=yourusername
set KAGGLE_KEY=yourapikey
```

### Option 2: Interactive Setup

Use the secure helper script to set up credentials interactively:

```bash
python secure_kaggle.py --setup
```

### Option 3: Manual Setup

1. Create a `.kaggle` directory in your home folder
2. Place your `kaggle.json` file there with proper permissions

## Running on GitHub

When running on GitHub (or any CI/CD environment):

1. Add `KAGGLE_USERNAME` and `KAGGLE_KEY` as repository secrets
2. Reference these secrets in your GitHub Actions workflow
3. The scripts will automatically detect and use them

## Results

The analysis will generate:

1. **Artifacts**: Trained models and vectorizers
2. **Visualizations**: Various plots showing bias metrics
3. **Reports**: Summary of bias metrics for each model

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

- [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
- [Conversation AI Research](https://conversationai.github.io/)

---

## Original Repository Information

This repository is based on the 3rd place solution by F.H.S.D.Y. of the Jigsaw Unintended Bias in Toxicity Classification competition.

Please see https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/97471#latest-582610 for more information.

### Original Requirements

```
apex
attrdict==2.0.1
nltk==3.4.4
numpy==1.16.4
optuna==0.13.0
pandas==0.24.2
pytorch-pretrained-bert==0.6.2
scikit-learn==0.21.2
torch==1.1.0
tqdm==4.32.1
```

Please refer to the original README for more information on configuration and execution of the original models.
