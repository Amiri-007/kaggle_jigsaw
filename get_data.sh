#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="$REPO_DIR/data"
mkdir -p "$DATA_DIR"

# ---------------------------------------------------------------------
# 1. Make sure Kaggle token exists
# ---------------------------------------------------------------------
if [[ ! -f ~/.kaggle/kaggle.json ]]; then
  echo "First-time setup – paste the **contents** of kaggle.json then press <Ctrl-D> 👇"
  mkdir -p ~/.kaggle
  cat > ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
  echo "✅ Saved token → ~/.kaggle/kaggle.json"
fi

# ---------------------------------------------------------------------
# 2. Skip if we already have train.csv
# ---------------------------------------------------------------------
if [[ -f "$DATA_DIR/train.csv" ]]; then
  echo "📝 Dataset already present → $DATA_DIR (skipping download)"
  exit 0
fi

# ---------------------------------------------------------------------
# 3. Download & unzip (≈ 720 MB)
# ---------------------------------------------------------------------
echo "📥  Downloading Civil Comments ..."
kaggle competitions download \
  -c jigsaw-unintended-bias-in-toxicity-classification \
  -p "$DATA_DIR"

echo "📦  Extracting ..."
unzip -q "$DATA_DIR/jigsaw-unintended-bias-in-toxicity-classification.zip" -d "$DATA_DIR"
echo "🎉  Dataset ready in $DATA_DIR" 