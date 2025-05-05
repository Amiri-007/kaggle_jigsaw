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
git clone https://github.com/yourusername/jigsaw-bias-audit.git
cd jigsaw-bias-audit
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

## Project Structure

```
.
├── artifacts/                # Saved model artifacts (vectorizers, models)
├── data/                     # Dataset files (downloaded via setup)
├── figs/                     # Output figures and visualizations
├── logs/                     # Training logs and validation results
├── models/                   # Trained model checkpoints
├── bias_metrics.py           # Metrics for bias evaluation
├── plot_utils.py             # Visualization utilities
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

## Local Setup for RTX 3070Ti & i7-12800HX

This project is optimized to run on a system with:
- NVIDIA RTX 3070Ti GPU
- Intel i7-12800HX CPU
- Windows 10/11

### Prerequisites

- Python 3.8+ (recommended: Python 3.10)
- NVIDIA CUDA 11.7+ and cuDNN (for GPU acceleration)
- [Kaggle API credentials](https://github.com/Kaggle/kaggle-api) (for dataset download)

### Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create and activate a virtual environment**:
   ```
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. **Install required packages**:
   ```
   pip install -r requirements.txt
   ```

4. **Setup Kaggle API**:
   - Go to your [Kaggle account settings](https://www.kaggle.com/account)
   - Click "Create New API Token" to download `kaggle.json`
   - Run the setup script with your kaggle.json file:
     ```
     python setup_environment.py --kaggle_json path/to/kaggle.json --data_dir ./data
     ```

5. **Start Jupyter notebook**:
   ```
   jupyter notebook
   ```

6. **Open and run the notebook**:
   - Open `Jigsaw_Unintended_Bias_Audit.ipynb` in Jupyter
   - Run all cells to perform the analysis

## Project Structure

```
.
├── data/                       # Dataset storage (created by setup script)
├── logs/                       # Output logs and validation results
├── artifacts/                  # Saved models and vectorizers
├── requirements.txt            # Python dependencies
├── setup_environment.py        # Environment setup script
├── Jigsaw_Unintended_Bias_Audit.ipynb  # Main analysis notebook
└── README.md                   # This file
```

## Models

The analysis compares two models:
1. **TF-IDF + Logistic Regression**: A baseline model using TF-IDF feature extraction
2. **BERT**: A pre-trained transformer model fine-tuned for toxicity detection

Both models are evaluated for unintended bias across different demographic groups.

## GPU Optimization

The notebook is specifically optimized for the RTX 3070Ti GPU:
- Uses mixed precision (FP16) for faster computation
- Enables TF32 acceleration for matrix operations
- Adjusts batch sizes to fit in GPU memory
- Uses PyTorch's CUDA optimizations

## Troubleshooting

**Out of Memory (OOM) errors**:
- Reduce the `batch_size` in the BERT inference section
- Reduce the number of rows loaded from the dataset (`nrows` parameter)

**CUDA errors**:
- Update your NVIDIA drivers to the latest version
- Make sure you have CUDA 11.7+ installed

## 📋 Project Overview

The project analyzes the [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) dataset to:

1. Evaluate baseline models (TF-IDF + LogReg, BERT) for toxicity classification
2. Analyze model performance across different demographic subgroups
3. Identify potential biases in the models' predictions
4. Visualize disparities in model performance

## 🛠️ Requirements

- Python 3.7+
- Kaggle account and API credentials
- ~4GB free disk space for the dataset
- GPU recommended for BERT model training/inference
- Windows, macOS, or Linux operating system

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone [repository-url]
cd jigsaw-bias-audit
```

### 2. Setup the environment

#### Windows:
```bash
setup.bat
```

#### macOS/Linux:
```bash
./setup.sh
```

The setup scripts will:
- Create a Python virtual environment
- Install the required dependencies
- Set up the directory structure
- Guide you through setting up the Kaggle API
- Download the dataset (if Kaggle credentials are configured)

### 3. Manual Setup (Alternative)

If the automated setup doesn't work, follow these steps:

1. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure Kaggle API:
   - Create a Kaggle account if you don't have one
   - Go to your account settings at https://www.kaggle.com/settings
   - Click "Create New API Token" to download kaggle.json
   - Place kaggle.json in ~/.kaggle/ and set permissions:
     ```bash
     # Windows
     mkdir -p %USERPROFILE%\.kaggle
     copy kaggle.json %USERPROFILE%\.kaggle\
     
     # macOS/Linux
     mkdir -p ~/.kaggle
     cp kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

4. Run the setup script:
   ```bash
   python setup_environment.py
   ```

## 📊 Running the Analysis

1. Activate the virtual environment (if not already activated):
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

2. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. Open and run `Jigsaw_Unintended_Bias_Audit.ipynb`

4. Alternatively, run the analysis script:
   ```bash
   python run_analysis.py
   ```

## 📂 Project Structure

```
.
├── artifacts/              # Saved model artifacts (vectorizers, models)
├── data/                   # Dataset files
├── figs/                   # Output figures and visualizations
├── logs/                   # Training logs and validation results
├── models/                 # Trained model checkpoints
├── bias_metrics.py         # Metrics for bias evaluation
├── plot_utils.py           # Visualization utilities
├── requirements.txt        # Python dependencies
├── run_analysis.py         # Script to run analysis
├── setup_environment.py    # Environment setup script
├── setup.bat               # Windows setup script
├── setup.sh                # Unix setup script
└── Jigsaw_Unintended_Bias_Audit.ipynb  # Main notebook
```

## 📈 Key Metrics

The project calculates several metrics to evaluate model bias:

- **Overall AUC**: Area Under the ROC Curve for the entire dataset
- **Subgroup AUC**: AUC calculated only on examples from a specific identity group
- **BPSN (Background Positive, Subgroup Negative) AUC**: Measures whether the model is more likely to give false positives to the subgroup than to the background
- **BNSP (Background Negative, Subgroup Positive) AUC**: Measures whether the model is more likely to give false negatives to the subgroup than to the background
- **Power Difference**: Ratio of the true positive rate of subgroup to the background

## 🔍 Interpreting Results

- **Equal AUC across subgroups**: Model performs similarly for all demographic groups
- **Lower Subgroup AUC**: Model performs worse on specific demographic groups, indicating potential bias
- **Lower BPSN**: Model gives more false positives to subgroup members
- **Lower BNSP**: Model gives more false negatives to subgroup members
- **Power Difference > 1**: Subgroup is more likely to be classified as toxic

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- [Jigsaw Unintended Bias in Toxicity Classification Competition](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
- [Original project by Michael and Julius for Responsible Data Science, Spring 2025]() 