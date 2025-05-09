#!/usr/bin/env python
"""
Cache tokenised encodings for transformer models to speed up multiple epochs.
"""
import argparse, pandas as pd, torch, pathlib, json
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm

def encode(df, tokenizer, max_len, out_path):
    all_ids, all_attn = [], []
    for txt in tqdm(df["comment_text"].astype(str).tolist(), desc=f"Tokenising {out_path.name}"):
        enc = tokenizer(txt,
                        max_length=max_len,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt")
        all_ids.append(enc["input_ids"][0])
        all_attn.append(enc["attention_mask"][0])
    torch.save({"ids": torch.stack(all_ids),
                "attn": torch.stack(all_attn)}, out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--csv", default="data/train.csv")
    ap.add_argument("--out_dir", default="data/tokenised")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Handle GPT-2 tokenizer pad token
        if "gpt2" in model_name and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set pad_token to eos_token for {model_name}")
            
        out_path = out_dir / f"{model_name.replace('/', '_')}_{args.max_len}.pt"
        if not out_path.exists():
            encode(df, tokenizer, args.max_len, out_path)
        else:
            print(f"Cached encoding exists: {out_path}")

if __name__ == "__main__":
    main() 