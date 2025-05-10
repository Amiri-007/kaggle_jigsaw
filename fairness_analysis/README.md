# Fairness Analysis Module

This module provides comprehensive tools for auditing and analyzing fairness in toxicity classification models across demographic groups. It implements various fairness metrics, bias evaluation techniques, and visualization tools to identify and quantify disparate impacts.

## Key Components

### Fairness Analysis Pipeline

- `run_fairness_analysis.py`: Complete fairness analysis pipeline
  - Runs demographic distribution analysis
  - Executes fairness auditing with metrics calculation
  - Performs intersectional fairness analysis
  - Checks compliance with fairness requirements
  - Launches the interactive fairness dashboard

### Fairness Metrics and Evaluation

- `metrics_v2.py`: Core implementation of fairness metrics
  - Selection rates and demographic parity calculations
  - Group-level disparity measures
  - Confusion matrix derived metrics (FPR, FNR disparities)
  - Kaggle-specific bias metrics (Subgroup AUC, BPSN, BNSP)

- `run_sharp_analysis.py`: SHarP individual fairness analysis
  - Analyzes differences in SHAP attribution patterns across demographic groups
  - Computes divergence scores showing attribution differences
  - Identifies potential reasoning disparities in the model

### Additional Tools

- `shap_report.py` and `shap_report_simple.py`: Generate fairness reports combining SHAP insights with fairness metrics

## Usage

```bash
# Run the complete fairness analysis pipeline
python fairness_analysis/run_fairness_analysis.py --model your_model_name

# Run specific fairness auditing
python fairness_analysis/audit_fairness_v2.py --preds results/your_model_preds.csv --val data/your_validation.csv --thr 0.6 --majority white

# Run SHarP individual fairness analysis
python fairness_analysis/run_sharp_analysis.py --sample 2000
```

## Fairness Metrics

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

## Interactive Dashboard

The fairness analysis pipeline includes an interactive dashboard built with Streamlit:

```bash
streamlit run scripts/fairness_dashboard.py
```

The dashboard provides:
- Overview of demographic distribution
- Interactive fairness metrics visualization
- Comparison of model performance across groups
- Threshold analysis for different fairness criteria
- Demographic parity and error rate disparity plots

## Output Files

Fairness analysis outputs are saved to various directories:
- `output/metrics/`: CSV files containing fairness metrics
- `figs/fairness/`: Visualizations of group disparities
- `figs/fairness_v2/`: Enhanced visualizations
- `figs/intersectional/`: Intersectional fairness analysis plots

## References

- [Fairness Definitions Explained](https://fairware.cs.umass.edu/papers/Verma.pdf)
- [Fairness and Machine Learning](https://fairmlbook.org/)
- [Aequitas Bias and Fairness Audit Toolkit](https://github.com/dssg/aequitas)

## Available Tools

### 1. Data Analysis

- `count_people.py`: Counts the number of comments and annotators across identity subgroups
- `count_people_viz.py`: Visualizes the distribution of data across identity subgroups
- `eda_identity.py`: Performs exploratory data analysis on identity columns

### 2. Fairness Metrics

- `audit_fairness_v2.py`: Comprehensive fairness audit tool that calculates:
  - Selection rates by demographic group
  - False Positive Rates (FPR) and False Negative Rates (FNR)
  - Demographic parity difference and ratios
  - Disparities relative to majority group
  - Violations of the 80% rule (0.8-1.2 range)

- `bias_auc_metrics.py`: Calculates specialized AUC metrics for bias assessment:
  - Subgroup AUC: Model performance within each demographic group
  - BPSN (Background Positive, Subgroup Negative): Measures model's ability to distinguish between background positives and subgroup negatives
  - BNSP (Background Negative, Subgroup Positive): Measures model's ability to distinguish between background negatives and subgroup positives

### 3. Compliance & Verification

- `check_compliance.py`: Verifies all required fairness analysis artifacts exist
  - Checks for presence of visualizations, metrics reports, and key analyses
  - Generates a Markdown compliance report showing completed items

### 4. Visualization & Reporting

- `fairness_dashboard.py`: Interactive Streamlit dashboard for exploring fairness metrics
  - Visualize key metrics across demographic groups
  - Compare multiple models
  - Filter by disparity thresholds
  - Check compliance with fairness requirements

## How to Use

### Running Fairness Audit

```bash
# Basic audit with default parameters
python scripts/audit_fairness_v2.py

# Specify custom parameters
python scripts/audit_fairness_v2.py --preds results/your_model_preds.csv --val data/your_validation.csv --thr 0.6 --majority white
```

### Launching the Dashboard

```bash
# Launch interactive fairness dashboard
streamlit run scripts/fairness_dashboard.py
```

### Checking Compliance

```bash
# Generate compliance report
python scripts/check_compliance.py
```

## Fairness Metrics Definitions

### Demographic Parity

Demographic parity requires that the probability of a positive prediction should be the same across all demographic groups. 

- **Demographic Parity Difference**: The difference between a group's selection rate and the overall selection rate
- **Demographic Parity Ratio**: The ratio of a group's selection rate to the overall selection rate

### Error Rate Parity

Error rate parity requires that error rates should be similar across different demographic groups.

- **FPR Disparity**: The ratio of a group's false positive rate to the majority group's false positive rate
- **FNR Disparity**: The ratio of a group's false negative rate to the majority group's false negative rate

### The 80% Rule (4/5 Rule)

A common fairness guideline that states that the selection rate for any group should be at least 80% of the selection rate for the group with the highest selection rate. We use a more symmetric definition where disparities should be within the range of 0.8 to 1.2.

## Fairness Considerations

When interpreting fairness metrics, consider:

1. **Intersectionality**: Different aspects of identity often intersect, and focusing on single demographic attributes may miss important patterns
2. **Context**: The importance of different fairness metrics depends on the specific context and consequences of model decisions
3. **Tradeoffs**: Different fairness metrics may be in tension with each other and with overall model performance
4. **Sample Size**: Groups with very few samples may have unreliable metrics

## Example Workflow

1. Explore data distribution with `count_people_viz.py`
2. Run basic fairness audit with `audit_fairness_v2.py` 
3. Check for compliance using `check_compliance.py`
4. Launch the dashboard for interactive exploration with `fairness_dashboard.py`
5. Address any identified fairness issues in the model 