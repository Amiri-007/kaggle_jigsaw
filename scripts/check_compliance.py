#!/usr/bin/env python
"""
Scan repo for required artefacts or code.  Produce YES/NO table in markdown.
"""
import pathlib, re, json, subprocess, sys, importlib, inspect, textwrap
from tabulate import tabulate

ROOT = pathlib.Path(".").resolve()
OUT  = pathlib.Path("output/compliance_report.md")
OUT.parent.mkdir(exist_ok=True)

# Helper -----------------------------------------------------------------------
def file_exists(pattern):
    return any(ROOT.glob(pattern))

def png_exists(name):
    return any(ROOT.glob(f"figs/**/*{name}*.png"))

def func_in_file(fname, pattern):
    p = ROOT / fname
    if not p.exists(): return False
    return bool(re.search(pattern, p.read_text()))

def header(txt, level=2):
    return "#"*level + " " + txt

rows = []

# 1. Exploratory Analysis ------------------------------------------------------
rows.append(["Exploratory: demo columns filtered",
             file_exists("scripts/eda_identity.py")])

rows.append(["Identity histograms",
             png_exists("identity_counts")])

rows.append(["Toxicity distribution plot",
             png_exists("toxicity_hist")])

rows.append(["Pairwise correlation heat-map",
             png_exists("corr_heatmap")])

rows.append(["Value counts printed (eda_summary.csv)",
             (ROOT/"output/eda_summary.csv").exists()])

# 2. Validate model performance -----------------------------------------------
rows.append(["Overall AUC computed",
             file_exists("results/summary.tsv")])

# 3. Bias AUC plots ------------------------------------------------------------
rows.append(["Subgroup AUC bar plot",
             png_exists("subgroup_auc")])
rows.append(["BPSN plot",
             png_exists("bpsn_auc_bar")])
rows.append(["BNSP plot",
             png_exists("bnsp_auc_bar")])

# 4. Confusion matrices --------------------------------------------------------
rows.append(["Overall confusion matrix fig",
             png_exists("conf_matrix")])

rows.append(["Per-subgroup confusion matrices",
             len(list(ROOT.glob("figs/fairness_v2/conf_matrix_*"))) > 0])

# 5. FPR / FNR per subgroup ----------------------------------------------------
rows.append(["FPR disparity plot",
             png_exists("fpr_disparity")])
rows.append(["FNR disparity plot",
             png_exists("fnr_disparity")])

# 6. Selection rate analysis ---------------------------------------------------
rows.append(["Selection-rate bar plot",
             png_exists("selection_rate")])

# 7. Demographic Parity diff & ratio ------------------------------------------
rows.append(["DP difference plot",
             png_exists("dp_difference")])
rows.append(["DP ratio plot",
             png_exists("dp_ratio")])

# 8. FPR / FNR disparity outside 0.8–1.2 flagged ------------------------------
summary_csv = ROOT/"output/fairness_v2_summary.csv"
flagged = False
if summary_csv.exists():
    import pandas as pd
    df = pd.read_csv(summary_csv)
    flagged = ((df["fpr_disparity"] < 0.8) | (df["fpr_disparity"] > 1.2) |
               (df["fnr_disparity"] < 0.8) | (df["fnr_disparity"] > 1.2)).any()
rows.append(["Disparities flagged 0.8-1.2",
             flagged])

# 9. AIF360 import presence ----------------------------------------------------
try:
    import importlib
    aif = importlib.util.find_spec("aif360")
    rows.append(["aif360 available", aif is not None])
except Exception:
    rows.append(["aif360 available", False])

# ----------------------------------------------------------------------------- #
report_md = []
report_md.append(header("Compliance Report"))
report_md.append("")
report_md.append(tabulate(rows, headers=["Item", "Done (✔/✘)"], tablefmt="github"))
OUT.write_text("\n".join(report_md), encoding="utf-8")
print("\n".join(report_md))
print(f"\n✅  Written to {OUT}") 