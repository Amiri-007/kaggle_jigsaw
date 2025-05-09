# Metric Cookbook (dev run)

Metric | Definition | Rubric tie-in
-------|------------|--------------
Overall AUC | ROC-AUC on entire validation set | _Accuracy of ADS_ (rubric §4a)
Subgroup AUC | AUC on rows with identity≥0.5 | Checks disparate true-positive/false-positive balance (§4b)
BPSN AUC | AUC on **B**ackground-Positive & **S**ubgroup-Negative | Detects false-positive bias (§4b)
BNSP AUC | AUC on **B**ackground-Negative & **S**ubgroup-Positive | Detects false-negative bias (§4b)
Generalised power-mean (p=-5) | Aggregates bias-metrics weighting worst cases | Used in Jigsaw competition; highlights worst subgroups (§4b)
FPR gap | |FPR(subgroup)–mean(FPR)| at τ=0.5 | Error-rate parity (§4b)
FNR gap | same for false-negatives | —
Final score | 0.25·overall AUC + 0.75·power-mean | Single headline fairness-aware score (§4a,b) 