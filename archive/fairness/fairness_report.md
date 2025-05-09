# Fairness Evaluation Report for DistilBERT Toxicity Classifier

## 1. Introduction

This report presents a comprehensive fairness evaluation of our DistilBERT-based toxicity classification model, trained on the Jigsaw Unintended Bias in Toxicity Classification dataset. The model was trained using a simplified pipeline while preserving the original architecture. This analysis examines how the model performs across different demographic groups.

## 2. Overall Model Performance

The model achieves a strong overall AUC of **0.953**. This indicates good general performance on the toxicity classification task.

![ROC Curve](figs/roc_distilbert_simplest.png)
*Figure 1: ROC curves for the overall model and best/worst performing demographic groups*

## 3. Demographic Group Performance

### 3.1 Subgroup AUC Comparison

The figure below shows the AUC scores for each demographic group:

![Subgroup AUC Comparison](figs/subgroup_auc_comparison.png)
*Figure 2: AUC comparison across demographic groups*

**Key observations:**
- Best performing group: atheist (AUC: 0.959)
- Worst performing group: female (AUC: 0.956)
- The maximum difference between demographic groups is only 0.003, indicating relatively consistent performance

### 3.2 Demographic Group Representation

The distribution of demographic groups in the validation dataset:

![Demographic Group Sizes](figs/demographic_group_sizes.png)
*Figure 3: Size of demographic groups in validation data*

**Key observations:**
- All demographic groups have substantial representation (14,000+ examples each)
- The representation is relatively uniform across groups, ensuring reliable fairness metrics

## 4. Fairness Metrics Analysis

### 4.1 Multiple Fairness Metrics

The performance of the model across different fairness metrics:

![Fairness Metrics Comparison](figs/fairness_metrics_comparison.png)
*Figure 4: Comparison of different fairness metrics across demographic groups*

**Key observations:**
- Subgroup AUC: Consistently high across all groups (0.956-0.959)
- BPSN AUC (Background Positive, Subgroup Negative): Good performance (0.955-0.959)
- BNSP AUC (Background Negative, Subgroup Positive): Lower performance (0.932-0.938)

### 4.2 Metric Heatmap

A detailed view of all metrics across groups:

![Fairness Metrics Heatmap](figs/fairness_metrics_heatmap.png)
*Figure 5: Heatmap of fairness metrics across demographic groups*

### 4.3 Bias Gap Analysis

The "bias gap" (difference between subgroup AUC and BNSP AUC) provides insight into where the model shows higher bias:

![Bias Gap](figs/bias_gap.png)
*Figure 6: Bias gap by demographic group*

**Key observations:**
- All groups have a positive bias gap, indicating BNSP AUC is consistently lower than subgroup AUC
- Larger gaps suggest more challenge in correctly classifying toxic content mentioning identity groups
- Christian and Hindu groups show the largest bias gaps

## 5. Representation Analysis

### 5.1 Positive Rate Comparison

The comparison of toxicity rates in demographic groups versus background:

![Positive Rate Comparison](figs/positive_rate_comparison.png)
*Figure 7: Comparison of positive (toxic) rates between subgroups and background*

**Key observations:**
- Most demographic groups have lower toxicity rates than their corresponding backgrounds
- This pattern suggests differences in content distribution that could impact model fairness

### 5.2 Performance Summary

A comprehensive summary of both performance and bias metrics:

![Fairness Summary](figs/fairness_summary.png)
*Figure 8: Summary of subgroup performance and bias gap*

**Key observations:**
- The top panel shows subgroup AUC with differences from overall performance
- The bottom panel shows the bias gap for each group
- This visualization highlights both absolute performance and bias concerns in a single view

## 6. Recommendations

Based on the fairness analysis, the following recommendations can help improve model fairness:

1. **Address BNSP performance gap**
   - Focus on improving the model's ability to correctly classify toxic content that mentions identity groups
   - Consider specialized data augmentation or training techniques

2. **Focus on female demographic group**
   - While the performance difference is small, targeted improvements for the female group can enhance overall fairness

3. **Investigate Christian and Hindu group bias**
   - These groups show the largest bias gaps and may benefit from specific fairness interventions

4. **Implement bias mitigation techniques**
   - Consider adversarial debiasing or post-processing calibration
   - Test fairness constraints during training

## 7. Conclusion

The DistilBERT model demonstrates strong overall performance with relatively small variations across demographic groups. The consistent AUC scores across different demographics (range: 0.956-0.959) suggest the model is reasonably fair in its predictions.

However, the lower BNSP scores indicate a common challenge in toxicity models: differentiating between toxic content about identity groups and benign mentions of those groups. This pattern is consistent across all demographic categories and represents the primary fairness challenge.

The model maintains a good balance between performance and fairness while being computationally efficient, making it suitable for production deployment with the recommended fairness improvements. 