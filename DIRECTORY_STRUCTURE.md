# Project Directory Structure

This document explains the organization of the project codebase for the Toxicity Classification System.

## Main Directories

- **src/**: Core source code for the toxicity classification system
- **configs/**: Configuration files for different model architectures and training scenarios
- **data/**: Data files and preprocessing utilities
- **pipelines/**: Scripts for running complete training and evaluation pipelines
  - `run_turbo.ps1`: Main turbo mode pipeline script
  - `run_turbo_simple.ps1`: Simplified turbo mode pipeline
  - `run_large_pipeline.ps1`: Pipeline for larger dataset experiments
- **predictions/**: Standalone prediction scripts
  - `run_custom_predict.py`: Custom prediction script
  - `large_predict.py`: Prediction script for larger datasets
- **explainers/**: Model explainability tools
  - `run_simplified_explainer.py`: Token attribution analysis
  - `generate_mock_shap.py`: SHAP value generation
  - `explainers_distilbert.py`: DistilBERT-specific explainers
- **output/**: Output files from model runs and analysis
- **results/**: Results of experiments and analysis

## Archive

Infrequently used scripts and tools have been moved to the archive directory:

- **archive/scripts/**: Archived utility scripts not used in the main pipeline
- **archive/analysis/**: Analysis tools that are not part of the core workflow
- **archive/fairness/**: Fairness metrics and analysis tools
- **archive/misc/**: Miscellaneous utilities and helpers

## Other Directories

- **docs/**: Documentation files
- **figs/**: Generated figures and visualizations
- **notebooks/**: Jupyter notebooks for interactive analysis
- **tests/**: Test scripts and testing utilities
- **legacy/**: Legacy code kept for reference 