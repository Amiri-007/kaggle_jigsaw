# Model Explainability Module

This module contains tools for analyzing and explaining model predictions using SHAP (SHapley Additive exPlanations) and other interpretability techniques. The explainability tools help understand how the model makes decisions and evaluate fairness at the feature attribution level.

## Key Components

### SHAP Analysis Tools

- `run_turbo_shap.py`: Comprehensive SHAP analysis for the turbo model
  - Generates SHAP values for model predictions
  - Creates waterfall plots, summary plots, and decision plots
  - Identifies the most influential tokens for toxicity classification

- `run_turbo_shap_simple.py`: Simplified token importance visualization
  - Uses token occlusion to identify important words
  - Generates color-coded text highlighting for toxicity explanations
  - Creates token importance bar charts

- `run_simplified_explainer.py`: Alternative token attribution analysis
  - Provides lightweight explanations with faster execution
  - Suitable for production environments

- `explainers_distilbert.py`: DistilBERT-specific explainers
  - Core implementation of SHAP explainers for the DistilBERT model

### Individual Fairness Analysis 

- Individual fairness analysis has been moved to the `fairness_analysis` module under `run_sharp_analysis.py`
  - The SHarP (SHAP-based fairness) analysis compares how attribution patterns differ across demographic groups
  - Generates divergence scores showing which groups have the most different reasoning patterns

## Usage

```python
# Run SHAP analysis on the turbo model
python explainability/run_turbo_shap.py --ckpt output/checkpoints/distilbert_headtail_fold0.pth

# Run simplified token importance analysis
python explainability/run_turbo_shap_simple.py --ckpt output/checkpoints/distilbert_headtail_fold0.pth

# Run the alternative explainer
python explainability/run_simplified_explainer.py --ckpt output/checkpoints/distilbert_headtail_fold0.pth
```

## Output Files

All explainability outputs are saved to the `output/explainability/` directory, including:
- SHAP values (.npy files)
- Token importance visualizations (.png, .html)
- Token attribution summaries (.csv)

## References

- [SHAP: SHapley Additive exPlanations](https://github.com/slundberg/shap)
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) 