#!/usr/bin/env python
"""
Count:
  (1) how many comment rows our model predicted on   (N_preds)
  (2) how many unique comment IDs exist in train     (N_train)
  (3) how many unique annotators contributed labels (N_annotators)
Outputs: stdout log + output/people_counts.csv
"""
import argparse, pathlib, pandas as pd, json, os, sys
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="results/preds_distilbert_dev.csv")
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--tox-anno", default="data/toxicity_individual_annotations.csv")
    ap.add_argument("--id-anno",  default="data/identity_individual_annotations.csv")
    ap.add_argument("--out", default="output/people_counts.csv")
    args = ap.parse_args()

    preds_df  = pd.read_csv(args.preds, usecols=["id"])
    train_df  = pd.read_csv(args.train, usecols=["id"])

    counts = {
        "n_pred_rows"   : len(preds_df),
        "unique_ids_pred": preds_df["id"].nunique(),
        "unique_ids_train": train_df["id"].nunique()
    }

    # Annotator counts (if files exist)
    for fkey, path in [("tox", args.tox_anno), ("id", args.id_anno)]:
        if pathlib.Path(path).exists():
            anno = pd.read_csv(path, usecols=["worker"])
            counts[f"unique_annotators_{fkey}"] = anno["worker"].nunique()
        else:
            counts[f"unique_annotators_{fkey}"] = "NA"

    # save & print
    out_p = pathlib.Path(args.out); out_p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([counts]).to_csv(out_p, index=False)
    print(json.dumps(counts, indent=2))

if __name__ == "__main__":
    main() 