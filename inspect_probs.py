"""
Inspect raw (pre-threshold) NSFW probability scores on the unsafe-diffusion
prompt set, broken down by source. Helps distinguish:
  - a THRESHOLD/CALIBRATION problem (scores are moderate, e.g. 0.3-0.55,
    just under the cutoff) from
  - a REPRESENTATION problem (scores are near-zero, meaning the model
    genuinely doesn't recognize the text as unsafe at all).

Usage:
  python inspect_probs.py \
    --model_path ./distilbert-base-uncased-nsfw/best \
    --data_csv /workspace/pinot/data/usafe_diffusion/all_prompts.csv \
    --calibration_file ./distilbert-base-uncased-nsfw/best/calibration.json
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--data_csv", type=str, required=True)
    p.add_argument("--calibration_file", type=str, default=None,
                   help="Optional calibration.json with 'temperature' and 'single_threshold' keys")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--n_examples", type=int, default=15,
                   help="Number of lowest-scoring false-negative examples to print per source")
    return p.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path).to(device)
    model.eval()

    id2label = model.config.id2label
    print(f"id2label: {id2label}")
    # Assume label id 1 = nsfw/unsafe (confirmed earlier for this pipeline)
    unsafe_id = 1

    temperature = 1.0
    threshold = 0.5
    if args.calibration_file:
        with open(args.calibration_file) as f:
            cal = json.load(f)
        temperature = cal.get("temperature", 1.0)
        threshold = cal.get("single_threshold", cal.get("optimal_threshold", 0.5))
        print(f"Loaded calibration: temperature={temperature:.4f}  threshold={threshold:.4f}")
    else:
        print("No calibration file given -- using raw softmax (T=1.0) and threshold=0.5")

    df = pd.read_csv(args.data_csv)
    print(f"Loaded {len(df)} rows | columns: {list(df.columns)}")

    texts = df["text"].tolist()
    true_labels = (df["safety_label"].str.lower() == "unsafe").astype(int).tolist()
    sources = df["source"].tolist()

    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i:i + args.batch_size]
            inputs = tokenizer(
                batch_texts, truncation=True, padding=True,
                max_length=args.max_length, return_tensors="pt"
            ).to(device)
            logits = model(**inputs).logits
            calibrated_logits = logits / temperature
            probs = torch.softmax(calibrated_logits, dim=-1)[:, unsafe_id].cpu().numpy()
            all_probs.extend(probs.tolist())
            if (i // args.batch_size) % 5 == 0:
                print(f"  [{min(i + args.batch_size, len(texts))}/{len(texts)}] processed")

    df["prob_unsafe"] = all_probs
    df["true_label"] = true_labels
    df["pred_label"] = (df["prob_unsafe"] >= threshold).astype(int)

    # ------------------------------------------------------------
    # Distribution summary by source, split by true label
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"PROBABILITY DISTRIBUTION BY SOURCE (threshold={threshold:.4f})")
    print("=" * 70)
    for src in df["source"].unique():
        sub = df[df["source"] == src]
        unsafe_sub = sub[sub["true_label"] == 1]
        if len(unsafe_sub) == 0:
            continue
        probs = unsafe_sub["prob_unsafe"]
        print(f"\n[{src}]  n_unsafe={len(unsafe_sub)}")
        print(f"  mean={probs.mean():.3f}  median={probs.median():.3f}  "
              f"min={probs.min():.3f}  max={probs.max():.3f}")
        # Bucket into ranges to see WHERE mass sits
        buckets = [0, 0.05, 0.2, 0.4, threshold, 0.8, 0.95, 1.0]
        counts, _ = np.histogram(probs, bins=buckets)
        for lo, hi, c in zip(buckets[:-1], buckets[1:], counts):
            bar = "#" * int(50 * c / max(len(unsafe_sub), 1))
            print(f"    [{lo:.2f}-{hi:.2f}): {c:4d}  {bar}")

    # ------------------------------------------------------------
    # Lowest-scoring false negatives per source (the clearest misses)
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"LOWEST-SCORING FALSE NEGATIVES (top {args.n_examples} per source)")
    print("=" * 70)
    fn = df[(df["true_label"] == 1) & (df["pred_label"] == 0)]
    for src in fn["source"].unique():
        sub = fn[fn["source"] == src].sort_values("prob_unsafe")
        print(f"\n--- {src} (n_false_neg={len(sub)}) ---")
        for _, row in sub.head(args.n_examples).iterrows():
            text_preview = str(row["text"])[:100].replace("\n", " ")
            print(f"  p={row['prob_unsafe']:.3f}  {text_preview!r}")

    out_path = "prob_inspection_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nFull results (with prob_unsafe column) saved to {out_path}")


if __name__ == "__main__":
    main()