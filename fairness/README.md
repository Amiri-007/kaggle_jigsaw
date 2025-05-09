# Fairness Analysis Tools

This module provides tools for auditing fairness in toxicity classification models, with specific focus on demographic subgroups.

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