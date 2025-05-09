# RDS Project: Deep Fairness for Toxicity Classification

This project provides a modern implementation for toxicity classification with deep learning models, focusing on bias reduction and fairness evaluation across demographic groups. It builds upon the [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) dataset to evaluate potential biases in machine learning models across demographic groups.

## Installation

```bash
# Clone the repository
git clone https://github.com/Amiri-007/kaggle_jigsaw.git
cd kaggle_jigsaw

# Create virtual environment and activate it
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (training, blending, fairness metrics, explanations)
make full-run

# For quick testing, run in dry-run mode
make full-run ARGS="--dry-run"
```

## Quick Start

```bash
# Train a model (BERT Head-Tail)
python -m src.train --model bert_headtail

# Run quick training test
python -m src.train --model bert_headtail --dry_run

# Full training with pseudo-labels + annotator weighting
python -m src.train --model bert_headtail --config configs/bert_headtail.yaml

# Generate pseudo-labels from trained model
python scripts/pseudo_label.py --base-model output/bert_headtail_fold0.pth --unlabeled-csv data/train.csv --out-csv output/pseudo_bert.csv

# Train with pseudo-labels (model improvement cycle)
python -m src.train --model bert_headtail --config configs/bert_headtail.yaml --pseudo-label-csv output/pseudo_bert.csv

# Generate predictions on test data
python -m src.predict --checkpoint_path output/bert_headtail_fold0.pth --test_file data/test_public_expanded.csv

# Generate fairness metrics and visualizations
python scripts/write_metrics.py --predictions output/preds/bert_headtail.csv
make figures
```

## Optional: Pre-Fine-Tuning on Old Data

This step is optional but can improve model performance (requires ~2h on an A100 GPU):

```bash
# Pre-finetune BERT on 2018 Jigsaw Toxic Comment dataset
python scripts/pre_finetune_old.py --model_name bert-base-uncased --mode classification --num_epochs 1
```

## Makefile Commands

```bash
# Show help
make help

# Train a model
make train

# Generate predictions
make predict CHECKPOINT=output/bert_headtail_fold0.pth

# Generate fairness figures
make figures

# Blend multiple model predictions
make blend GROUND_TRUTH=data/valid.csv
```

## Features

- State-of-the-art deep learning models for toxicity classification:
  - LSTM-Capsule with EMA (Exponential Moving Average)
  - BERT with head-tail architecture (processes first and last 128 tokens)
  - GPT-2 with head-tail architecture
- Advanced negative downsampling and weighted training
- Pseudo-labeling for semi-supervised learning
- Annotator count weighting to prioritize higher consensus examples
- Comprehensive fairness evaluation framework
- Interactive visualization dashboard for fairness metrics
- Optimal model blending with Optuna

## Project Structure

```
.
├── configs/                 # Model configuration files
├── data/                    # Dataset files (download from Kaggle)
├── fairness/                # Fairness evaluation framework
├── figs/                    # Output figures and visualizations
├── legacy/                  # Legacy code (for reference)
├── notebooks/               # Jupyter notebooks
├── output/                  # Model outputs (checkpoints, predictions)
│   └── preds/               # Model predictions
├── results/                 # Evaluation results (metrics, reports)
├── scripts/                 # Utility scripts
│   └── write_metrics.py     # Generate fairness metrics
├── src/                     # Source code
│   ├── data/                # Data loading and processing
│   ├── models/              # Model implementations
│   │   ├── lstm_caps.py     # LSTM-Capsule model
│   │   ├── bert_headtail.py # BERT head-tail model
│   │   └── gpt2_headtail.py # GPT-2 head-tail model
│   ├── train.py             # Training script
│   ├── predict.py           # Prediction script
│   └── blend_optuna.py      # Model blending optimization
└── tests/                   # Unit tests
```

## Models

The project implements three deep learning models:

### 1. LSTM-Capsule

- PyTorch implementation with nn.Embedding using GloVe embeddings
- Bidirectional LSTM encoder
- Primary Capsule layer with dimension=8
- Self-attention mechanism
- Exponential Moving Average (EMA) with decay=0.999

### 2. BERT Head-Tail

- Processes both the head (first 128 tokens) and tail (last 128 tokens) of the input
- Concatenates the [CLS] representations from both head and tail
- Uses HuggingFace Transformers library
- Linear warmup and decay learning rate schedule

### 3. GPT-2 Head-Tail

- Similar architecture to BERT Head-Tail but using GPT-2 as the base model
- Optimized for toxicity classification tasks

## Handling Long Sequences

Both BERT and GPT-2 models use a head-tail architecture to handle long sequences:
- First 128 tokens capture the beginning context
- Last 128 tokens capture the conclusion/resolution
- Two separate encodings are processed and then combined
- Helps capture context from both parts of long comments

## Data Sampling Strategy

The models use a sophisticated sampling strategy to address class imbalance:

- **Epoch 1**: Drop 50% of rows where target < 0.2 AND identity_columns.sum() == 0
- **Epoch 2+**: Restore half of the dropped examples

## Sample Weighting

Per-sample weights are calculated using:
```
w = 1
w += 3 * identity_sum
w += 8 * target
w += 2 * log(toxicity_annotator_count + 2)  # Weights examples with higher annotator agreement
w /= w.max()
```

This prioritizes:
1. Examples with identity mentions
2. Toxic examples
3. Examples with higher annotator agreement (higher confidence label)
4. While maintaining balanced training

## Pseudo-Labeling

The project implements a pseudo-labeling technique to leverage unlabeled data:

1. Train a base model on the labeled dataset
2. Use the base model to predict on unlabeled data (or training data for self-training)
3. Keep only high-confidence predictions (above 0.9 or below 0.1)
4. Add these examples with their predicted "pseudo-labels" to the training set
5. Retrain the model on the combined dataset

Benefits:
- Increases effective training data size
- Helps models learn from high-confidence examples
- Improves generalization, especially for underrepresented classes

The process can be repeated iteratively to further improve model performance, a common technique in semi-supervised learning.

## Fairness Evaluation

The project includes comprehensive fairness evaluation:

- Subgroup AUC across demographic groups
- BPSN (Background Positive, Subgroup Negative) AUC
- BNSP (Background Negative, Subgroup Positive) AUC
- Error rate metrics (FPR/FNR) at τ=0.5 threshold
- Threshold gap visualization

## Model Blending

Optimal weights for model blending are found using Optuna:

```bash
python -m src.blend_optuna --ground_truth data/valid.csv
```

This optimizes for the fairness metric rather than just classification accuracy.

## Continuous Integration

GitHub Actions workflow tests:
- Training (dry run with 5 mini-batches)
- Unit tests
- Figure generation

## License

MIT License

## References

- [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
- [Perspective API](https://perspectiveapi.com/)
- [Kaggle 3rd Place Solution]([https://medium.com/@yanpanlau/jigsaw-unintended-bias-in-toxicity-classification-top-3-solution-a1309ff8fc53](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/discussion/97471))
