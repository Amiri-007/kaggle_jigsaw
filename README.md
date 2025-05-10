# Fairness Audit of Jigsaw Toxicity Classifier

**Project**: Audit of unintended bias in toxic comment classification using the Jigsaw Civil Comments dataset. This repository contains the full implementation of a toxicity classification model and a comprehensive fairness evaluation, as described in the report *“Fairness Audit: Jigsaw Unintended Bias in Toxicity Classification.”* We train a state-of-the-art text toxicity detector and then assess its performance across demographic subgroups for bias, using both traditional fairness metrics and model explainability (SHAP values).

## Overview and Purpose

Online toxicity detection models can inadvertently behave unfairly toward certain protected groups (e.g., flagging non-toxic comments about specific races or religions as toxic). This project builds a **toxic comment classifier** and then performs a rigorous **fairness audit**. Key aspects include:

* **Model Implementation:** A BERT-based classification model (with comparisons to LSTM and GPT-2 variants) trained on the Civil Comments dataset.
* **Bias Metrics:** Evaluation of model bias using subgroup AUC metrics (from Jigsaw competition) and classic fairness measures like demographic parity and error rate parity.
* **Explainability:** Use of SHAP (Shapley Additive Explanations) to interpret model predictions and a custom “**SHarP**” score (SHAP-based fairness metric) to quantify how differently the model treats identity-related content.

The goal is to identify bias in the model’s behavior and suggest ways to mitigate such unintended biases. This repository is structured to allow anyone (or graders) to **reproduce our results and figures** and inspect the code for each component of the analysis.

## Repository Structure

```bash
├── model_impl/              # Implementation of models and training pipeline (formerly "src/")
│   ├── train.py             # Training script for models (handles data loading, training loop, evaluation)
│   ├── predict.py           # Script to generate predictions using a trained model
│   ├── blend_optuna.py      # (Optional) Blending script to ensemble multiple models with Optuna
│   ├── data/                # Data loading and preprocessing utilities
│   ├── models/              # Model architecture definitions 
│   │   ├── bert_headtail.py     # BERT (head-tail) model class
│   │   ├── lstm_caps.py         # LSTM with capsule network model class
│   │   └── gpt2_headtail.py     # GPT-2 (head-tail) model class
│   └── utils/              # Utility functions (e.g., text processing, downsampling)
│
├── fairness_analysis/       # Fairness evaluation scripts and metrics
│   ├── metrics_v2.py        # Metrics for bias: subgroup AUC, BPSN, BNSP, etc.
│   ├── audit_fairness_v2.py # Script to compute fairness metrics (selection rate, parity, FPR/FNR disparities)
│   ├── audit_accuracy_fairness.py  # Script to compute confusion matrix and accuracy + bias metrics together
│   ├── bias_auc_metrics.py  # (Optional) Script to compute Kaggle competition bias AUC metrics separately
│   ├── count_people.py      # Data analysis script for counting subgroup representation in dataset
│   └── fairness_dashboard.py# (Optional) Interactive dashboard for fairness (Streamlit app)
│
├── explainability/          # Model explainability and SHAP analysis
│   ├── run_turbo_shap.py        # Runs SHAP analysis on the turbo (DistilBERT) model
│   ├── run_turbo_shap_simple.py # Simplified token-occlusion importance analysis
│   ├── run_sharp_analysis.py    # **SHarP individual fairness** analysis (computes SHAP divergence for identity groups)
│   ├── generate_mock_shap.py    # Utility to generate dummy SHAP data (for testing)
│   └── shap_report.py           # Produces a report combining SHAP results with fairness insights
│
├── configs/                 # Configuration files for training models
│   ├── bert_headtail.yaml        # Config for full BERT model training (hyperparameters, file paths)
│   ├── bert_headtail_turbo.yaml  # Config for “turbo” (quick run) DistilBERT model
│   ├── lstm_caps.yaml           # Config for LSTM-Capsule model
│   └── gpt2_headtail.yaml       # Config for GPT-2 model
│
├── data/                    # Data directory (contains train.csv, test.csv after download)
│   └── README.md            # Notes on dataset (if needed, e.g., source of data)
│
├── notebooks/               # Jupyter notebooks for analysis and figure generation
│   ├── 04_generate_figures.ipynb  # Notebook to generate plots for the report
│   └── exploration_eda.ipynb      # Any EDA or exploratory analysis (if applicable)
│
├── output/                  # Output directory for all results (created after running pipelines)
│   ├── checkpoints/             # Saved model checkpoints (.pth files)
│   ├── predictions/             # Model prediction CSVs on validation/test sets
│   ├── metrics/                 # CSV or JSON files of computed metrics (e.g., bias report)
│   ├── figures/                 # Plots and figures (confusion matrix, bias bar charts, etc.)
│   └── explainability/          # SHAP outputs (shap values .npz, SHarP scores .csv, explanation visualizations)
│
├── pipelines/               # Pipeline scripts to run end-to-end processes (mostly for Windows PowerShell)
│   ├── run_full_pipeline.ps1    # Runs full data preparation, training, evaluation, and analysis
│   ├── run_turbo.ps1            # Runs a faster subset training pipeline (uses turbo configs, 5% data)
│   └── run_large.ps1            # Runs pipeline on a larger dataset sample (if needed, e.g., 8% data)
│
├── README.md                # **Main documentation** (you are reading this)
├── requirements.txt         # Python dependencies for the project
└── archive/                 # Archived code (old scripts, legacy analysis – not needed for main results)
    └── ...                  # (Contains older versions of fairness metrics, etc., for reference only)
```

*(The structure above assumes some renaming as recommended. In the actual repo, `model_impl` was `src`, and `explainability` was `explainers`. Adjust names according to the final structure.)*

## Installation and Setup

**Requirements:** Python 3.8+, PyTorch (tested on 1.9+), and an environment capable of running Jupyter notebooks (for generating figures). See `requirements.txt` for a full list of packages. Key libraries include `transformers` (HuggingFace), `shap`, `sklearn`, `pandas`, `numpy`, and `matplotlib/seaborn` for plotting.

1. **Clone the repository**

   ```bash
   git clone https://github.com/Amiri-007/kaggle_jigsaw.git
   cd kaggle_jigsaw
   ```

2. **Create a virtual environment and install dependencies**
   We recommend using a virtual environment to avoid version conflicts:

   ```bash
   python -m venv venv
   source venv/bin/activate         # On Linux/Mac
   # For Windows PowerShell, use: .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt  # Install required Python packages
   ```

3. **Download the dataset** – *Civil Comments (Jigsaw Unintended Bias)*:
   The training data is from a Kaggle competition. We provide a script to download it automatically. **You will need a Kaggle API token** (a `kaggle.json` file) for this to work.

   ```bash
   # One-time dataset download (prompts for Kaggle API credentials)
   make data
   ```

   This will download the dataset CSVs into the `data/` folder (around 1.9M comments in `train.csv`). Alternatively, you can manually download the **Jigsaw Unintended Bias in Toxicity Classification** data from Kaggle and place `train.csv` and `test.csv` in the `data/` directory.

4. **(Optional) Review Configurations:** The `configs/` folder contains YAML files for different model setups. The defaults should work out-of-the-box. You can adjust hyperparameters (like number of epochs, learning rate, or model type) in these files if needed. In particular, `*_turbo.yaml` configs use smaller subsets for quick experiments.

## How to Run the Model Training and Evaluation

The repository provides **Makefile commands and pipeline scripts** to streamline running the experiments. Here are the typical steps to reproduce the results:

### 1. Training the Model(s)

To train the toxicity classification model, you have two main options:

* **Full Training Run:** Train the main model on the full dataset and evaluate. This is resource-intensive but yields the exact results used in the report.

  ```bash
  make full-run
  ```

  This Makefile target will:

  * Train the BERT-headtail model on the full training set (can be adjusted to train multiple models, e.g., LSTM, GPT-2, if desired).
  * Save the trained model checkpoint to `output/checkpoints/`.
  * Generate validation predictions and save them to `output/predictions/` (e.g., `preds_bert_base.csv`).
  * Compute initial metrics and bias scores on the validation set, saving results to `output/metrics/`.
  * (Optional) Blend models if multiple models were trained, and output the blended predictions.

  *Notes:* By default, `make full-run` will use GPU and may take a few hours depending on hardware. Ensure your environment has a GPU available or adjust settings for CPU (training on CPU will be much slower). You can control training aspects via the config files or command-line flags (for example, to run fewer epochs or a smaller sample for testing).

* **Turbo (Quick) Run:** If you just want to do a quick end-to-end run (for example, to verify everything works or to generate sample outputs without waiting hours), you can run the *turbo mode* with a smaller subset of data:

  ```bash
  # Turbo pipeline: trains on 5% of data for quick results (~10-15 minutes)
  ./pipelines/run_turbo.ps1        # For Windows PowerShell users
  # OR (equivalent Python command for any OS)
  python model_impl/train.py --config configs/bert_headtail_turbo.yaml --model bert_headtail --fp16 --turbo
  ```

  The turbo pipeline will train a lightweight DistilBERT model on 5% of the data and perform subsequent steps quickly. It produces outputs in the same directories (but with “\_turbo” in filenames). Results won’t match the full report but are useful for testing the pipeline.

During training, logs will be printed to stdout (and saved in `output/` if configured). After training, you should have model file(s) in `output/checkpoints/` and prediction files in `output/predictions/`.

### 2. Evaluating Performance and Bias

Once you have a trained model and its predictions on a validation set, you can run the fairness audit to compute all relevant metrics:

* **Fairness Audit Script:** Use the bias audit script to compute subgroup performance and fairness metrics:

  ```bash
  python fairness_analysis/audit_fairness_v2.py --preds output/predictions/preds_bert_base.csv --val data/valid.csv --thr 0.5 --majority white
  ```

  Replace `preds_bert_base.csv` with the actual predictions file and `data/valid.csv` with the validation dataset used (in our setup, we often use a portion of `train.csv` as validation). This script will output:

  * Subgroup selection rates (what fraction of comments in each demographic got predicted “toxic”).
  * False positive rate (FPR) and false negative rate (FNR) for each subgroup.
  * Demographic parity difference and ratio for each subgroup compared to the overall population.
  * Disparity scores and an 80% rule check for each metric (marks if any group falls outside the 0.8 to 1.25 range relative to the majority).
  * A summary bias report (printed to console and saved to `output/metrics/fairness_report.csv` or similar).

  These metrics correspond to those in the report’s fairness analysis section. For example, the script will explicitly report if any subgroup’s positive prediction rate is much higher or lower than others (indicating potential bias).

* **Accuracy and Confusion Matrix Audit:** We also provide `audit_accuracy_fairness.py` which, given the predictions and true labels, will output confusion matrix stats (TP, FP, TN, FN) overall and possibly per subgroup, along with accuracy and balanced accuracy. You can run:

  ```bash
  python fairness_analysis/audit_accuracy_fairness.py --preds output/predictions/preds_bert_base.csv --val data/valid.csv --thr 0.5
  ```

  This will print overall model accuracy, precision, recall, as well as highlight any skew in errors among subgroups. (In our report, for instance, we noted very high FPR overall and especially for certain identities – this script helps pinpoint such issues.)

* **Bias AUC Metrics (Kaggle Competition Metric):** If you want to compute the “bias AUC” metrics (the ones used in the Jigsaw competition: Subgroup AUC, BPSN AUC, BNSP AUC and the combined score), use:

  ```bash
  python fairness_analysis/bias_auc_metrics.py --preds output/predictions/preds_bert_base.csv --val data/valid.csv
  ```

  This will output the AUC for each identity subgroup, the BPSN and BNSP AUCs, and the final weighted score. Our `fairness_analysis/metrics_v2.py` implements these calculations efficiently. These metrics were discussed in the report to evaluate performance disparities. The results will be saved (e.g., to `output/metrics/bias_auc_results.csv`).

All the above scripts log their output to the console and save CSV files under `output/metrics/` for record-keeping. After running them, you should have a collection of metrics quantifying the model’s fairness issues (for example, you might see that the model’s AUC on comments mentioning “black” is much lower than overall AUC, confirming a fairness concern raised in the report).

### 3. Model Explainability Analysis (SHAP)

Understanding *why* the model is making certain predictions is crucial for a fairness audit. We include several explainability tools:

* **Token Importance (Simplified):** To get a quick sense of which words the model finds most toxic, run:

  ```bash
  python explainability/run_turbo_shap_simple.py --ckpt output/checkpoints/bert_headtail_model.pth
  ```

  (Replace with your model checkpoint path.) This will generate:

  * A bar chart of top contributing tokens for toxicity.
  * A visualization of a few example comments with tokens highlighted (red for contributing to toxic label, blue for mitigating).
  * The outputs are saved in `output/explainability/` (e.g., `token_importance.png` and some `.csv` files of token contributions).

  This uses an occlusion-based approach (masking tokens) rather than full SHAP, so it’s fast and works with no heavy requirements.

* **Full SHAP Analysis:** For a more comprehensive interpretability analysis using SHAP:

  ```bash
  python explainability/run_turbo_shap.py --ckpt output/checkpoints/bert_headtail_model.pth
  ```

  This script loads the model and computes SHAP values for a sample of validation data (by default). It may take some time and memory, as SHAP for large models is computationally heavy. The output will be:

  * `shap_values.npz` containing raw SHAP arrays,
  * `shap_summary.png` or similar charts showing feature importance,
  * example-based explanations (if configured, it can save per-comment breakdowns).

  (If you encounter stability issues with SHAP, consider using the simplified approach above or reducing sample size.)

* **SHarP Individual Fairness Analysis:** This is the novel part where we connect SHAP with fairness:

  ```bash
  python explainability/run_sharp_analysis.py --ckpt output/checkpoints/bert_headtail_model.pth --sample 2000
  ```

  This will perform the **SHarP analysis** as described in the report. It does the following:

  1. Sample 2,000 comments from the validation set (you can adjust `--sample`).
  2. Compute SHAP values for each comment.
  3. Calculate the average SHAP attribution vector for comments that mention each identity (e.g., all comments where `male=1` vs those where `male=0`).
  4. Compute the cosine distance between each subgroup’s average SHAP vector and the overall average – this is the **SHarP divergence score** for that identity group.
  5. Save results to `output/explainability/`:

     * `sharp_scores.csv`: a table of divergence scores for each identity subgroup.
     * `sharp_divergence.png`: a bar chart of the divergence (as in the report’s figure, where higher bars indicate the model’s feature attributions behave very differently when that identity is present).
     * (Also saves raw shap values in `distilbert_shap_values.npz` for reference.)

  This analysis helps identify groups for which the model’s “reasoning” differs significantly. For example, we found in the report that the **“black”** and **“homosexual\_gay\_or\_lesbian”** subgroups had the highest SHarP scores, indicating the model’s token importance pattern for those comments was very different (likely focusing on the identity words themselves, which is a sign of bias). Running this script will allow you to reproduce that finding.

All explainability outputs (charts, .csv files) will be in the `output/explainability/` folder. We recommend checking the `sharp_divergence.png` and comparing the values in `sharp_scores.csv` to see which identities the model might be treating unfairly via its internal logic. These results tie directly into the fairness metrics above – often, high SHarP divergence correlates with high error rates or low AUC for that group, as noted in the report.

### 4. Generating Figures for the Report

We have provided a notebook `notebooks/04_generate_figures.ipynb` (and a Python script equivalent) that takes the metrics and outputs from above and generates the polished figures used in the report. To generate all figures in one go, you can run:

```bash
jupyter nbconvert --execute notebooks/04_generate_figures.ipynb --to html --ExecutePreprocessor.timeout=600
```

This will execute the notebook (which reads the metrics CSVs and output files) and produce an HTML with all plots. Key figures include:

* Confusion matrix visualization (showing false positive/negative rates).
* Bar charts of AUC scores per subgroup (subgroup AUC, BPSN, BNSP).
* Demographic parity plot (difference in selection rates).
* The SHarP divergence bar chart per identity.
* Token attribution examples.

The figures will also be saved as image files in `output/figures/` for convenience. You can refer to these images when reading the report to cross-verify the findings.

## Major Files and Their Purpose

To help navigate the repository, here is a quick explanation of important files/modules and what they do (organized by category):

* **Model Implementation (model\_impl/):**

  * `train.py`: Orchestrates model training. Handles loading the dataset, initializing the model and tokenizer (BERT, GPT-2, or LSTM based on `--model` argument), training for several epochs, and evaluating on validation data each epoch. It uses `models/*.py` classes for the actual model definitions and `data/loaders.py` for the dataset. Logging is enabled to track training loss and AUC. *Use:* `python model_impl/train.py --config configs/bert_headtail.yaml --model bert_headtail`.
  * `predict.py`: Loads a saved checkpoint and generates predictions on a given dataset (e.g., test set). This is used if you want to produce predictions for external test data or for additional analysis. *Use:* `python model_impl/predict.py --checkpoint_path output/checkpoints/your_model.pth --test_file data/test.csv`.
  * `models/bert_headtail.py`: Defines the BERT-based model architecture we used. It loads a pre-trained BERT model and adds a classification head. The “head-tail” mechanism means it takes a long text and feeds the first part and last part of the text to BERT (to handle very long comments). There are similar files for `lstm_caps` (an LSTM with capsule layers) and `gpt2_headtail` (GPT-2 with head-tail handling). These were experimented with; the report primarily focuses on BERT results.
  * `data/loaders.py`: Contains `ToxicDataset` class (subclass of `torch.utils.data.Dataset`) which prepares text and identity features for training. It tokenizes text using the chosen tokenizer and sets up tensors. It also applies negative downsampling if specified (to tackle label imbalance by sampling non-toxic comments).
  * `configs/*.yaml`: Configuration files that specify hyperparams like learning rate, batch size, number of epochs, model type, and file paths for train/val data. For example, `bert_headtail.yaml` might specify using `bert-base-uncased` as model and include training file names.

* **Fairness Analysis (fairness\_analysis/):**

  * `metrics_v2.py`: **Library of fairness metrics.** Defines functions to detect identity columns in the data and compute metrics: `subgroup_auc`, `bpsn_auc`, `bnsp_auc`, as well as demographic parity differences, and a final combined score. This is used under the hood by other scripts to get bias metrics efficiently.
  * `audit_fairness_v2.py`: **Main fairness auditing script.** Calculates group-wise selection rates, FPR/FNR, demographic parity, and 80% rule compliance. It takes predicted probabilities and a threshold to decide toxic/not toxic. Outputs a summary of which groups might be facing bias (e.g., if a group has much higher false positive rate than the majority, it will flag that). This script prints a detailed report and saves a CSV summary.
  * `audit_accuracy_fairness.py`: Combined evaluation script for overall performance and fairness. It will output the confusion matrix values and accuracy, as well as call bias metrics for subgroups. It’s useful for getting a one-shot view of everything after training a model.
  * `count_people.py` and `count_people_viz.py`: Utility scripts to examine the dataset. These count how many comments and annotators belong to each identity subgroup and can plot the distribution. (This can help understand data representation – an insight in the report was that some groups had very few examples, contributing to high error rates.)
  * `fairness_dashboard.py`: *Optional.* This is a Streamlit app that can visualize the fairness metrics in a web dashboard format. While not needed for grading, it shows our exploration tool – you can run `streamlit run fairness_analysis/fairness_dashboard.py` to launch an interactive UI to toggle between models and see metric charts.

* **Explainability (explainability/):**

  * `run_turbo_shap.py`: Uses the SHAP library to compute token attributions on model predictions. By default, it runs on the DistilBERT “turbo” model (hence the name) and uses a subset of data for speed. It requires `shap` package and a running environment that can handle it. The output includes SHAP value files and potentially a SHAP summary plot (global feature importance).
  * `run_turbo_shap_simple.py`: A simpler, robust approach to get token importances. It systematically masks each token in sample comments and measures the drop in predicted toxicity probability. It then visualizes the most important tokens. This was used to quickly sanity-check model behavior (for example, to confirm that identity words like “gay” or “black” strongly influence the model’s prediction – indicating bias).
  * `run_sharp_analysis.py` (formerly `run_individual_fairness.py`): **SHAP-based Fairness (SHarP) analysis.** As explained earlier, this computes the SHarP divergence scores. It loads the model, a sample of validation data, and goes through SHAP calculations to produce per-group attribution differences. It is the core script behind one of our report’s key fairness findings. The results help answer whether the model’s decision process is uniform or varies by demographic – a form of checking **individual fairness** (similar cases treated similarly).
  * `shap_report.py`: This script can generate a consolidated report by reading SHAP outputs and identity data. It can, for example, find which tokens are most contributing to toxicity for each subgroup or how the presence of certain identity terms shifts the SHAP value distribution. In practice, we used a combination of the above scripts and manual analysis in the notebook, but this script automates some of that. It outputs a markdown or HTML report that pairs well with the fairness metrics.
  * (Various output files in `output/explainability/`: after running explainability scripts, you’ll see images like `*_importance.png`, `sharp_divergence.png`, and data files like `*.npz` or `*.csv`. These are intermediate results that feed into the final analysis and report.)

* **Pipelines and others:**

  * `pipelines/run_full_pipeline.ps1`: A Windows PowerShell script that strings all the steps together – from data download to training to generating metrics and figures. It’s essentially a one-click solution for Windows users to reproduce everything. It calls the Python scripts in sequence and ensures outputs from one step feed into the next.
  * `Makefile`: Defines convenient commands (`make train`, `make predict`, `make audit`, etc.). Even without PowerShell, a combination of `make` targets can achieve the full run. For example, `make train` (trains model), then `make audit-v2` (runs fairness audit on results), `make sharp-fast` (runs a quick SHarP analysis). We included these for ease of use and to standardize common tasks.
  * `requirements.txt`: Python packages needed. Worth noting: ensure you have the correct version of SHAP (we used `shap==0.43.0`) and transformers. Using `pip install -r requirements.txt` should handle versioning.

## External Data and Assumptions

* **Data:** As noted, you need the Jigsaw Civil Comments dataset. The repository does not include the large CSVs due to size, but will download them with the provided script (or you can add them manually). We assume the dataset is the same format as the original competition (with identity columns like `male`, `female`, etc., already in the CSV). If using a custom or modified dataset, ensure the column names match those expected by our code (see `fairness_analysis/metrics_v2.py` for the list of identity columns used).
* **Pretrained Models:** Our model implementations rely on Hugging Face transformers for BERT and GPT-2. The code will automatically download the pretrained weights for `bert-base-uncased`, `distilbert-base-uncased`, and `distilgpt2` as needed (internet connection required for the first run to fetch these). We assume availability of these model weights.
* **System Requirements:** Training BERT on the full data ideally needs a GPU with \~12GB memory (we use gradient accumulation if needed, which can be configured in `configs/`). Disk space: the dataset is \~500MB, and outputs (models + shap values) can be a few 100 MBs. SHAP computations can be memory intensive; if you face memory issues, use the `--sample` flag to reduce data or the simplified explainer.
* **Randomness and Reproducibility:** We set seeds in the code for reproducibility. Minor differences in results (e.g., AUC fluctuations at the 3rd decimal place) might occur, but overall trends should be consistent. If you follow the exact pipeline, you should replicate the report’s numbers closely (our final model AUC, bias scores, etc., as documented).
* **Report Alignment:** Every figure or table in the report can be traced back to an output from this code. For instance, *Table 1: Bias AUC scores* comes from `bias_auc_metrics.py` output, *Figure X: SHAP divergence* comes from `run_sharp_analysis.py` output, etc. We aimed to make the code self-explanatory and modular, so one could investigate further (e.g., tweak the threshold in fairness audit or test the model on custom sentences).

## Conclusion

This repository is a self-contained package for auditing the fairness of a toxicity classification model. By following the steps above, you can train the model and reproduce the fairness evaluation presented in the report. The code is organized to highlight the **separation between model implementation and fairness analysis**, reflecting the structure of our study. We hope this project serves as a useful reference for fairness audits in NLP classification systems.

**Please feel free to reach out or open an issue** if you encounter any problems or have questions about running the code or understanding the analysis.

## License

MIT License

## References

- [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
- [Perspective API](https://perspectiveapi.com/)
- [Kaggle 3rd Place Solution](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/discussion/97471)
- [SHAP: SHapley Additive exPlanations](https://github.com/slundberg/shap)
