"""
Debug script: surface WildJailbreak false negatives and raw model scores.

A "false negative" here means a harmful prompt that the model scored below the
decision threshold (i.e. it *escaped* detection — the attack succeeded).

Outputs
-------
  - Console: score-bucket histogram, then per-example table of false negatives
             sorted by harmful_prob ascending (worst failures first).
  - CSV    : all examples with columns
               data_type | text | true_label | pred_label | harmful_prob | error_type
             where error_type ∈ {false_negative, false_positive, correct}

Usage
-----
  # Minimal — uses default threshold 0.5, no calibration
  python debug_wildjailbreak.py --model_path tsaxena/distilbert-nsfw

  # With calibration (recommended)
  python debug_wildjailbreak.py --model_path tsaxena/distilbert-nsfw \\
      --calibration_file tsaxena-distilbert-nsfw-calibration/calibration.json

  # Limit console output to top-N worst false negatives
  python debug_wildjailbreak.py --model_path tsaxena/distilbert-nsfw \\
      --calibration_file tsaxena-distilbert-nsfw-calibration/calibration.json \\
      --top_n 30 --output_file debug_predictions.csv
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

EVAL_DATASET = "allenai/wildjailbreak"
TEXT_PREVIEW_CHARS = 120


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Debug WildJailbreak false negatives and raw model scores",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Local path or HuggingFace hub ID of the fine-tuned model",
    )
    parser.add_argument(
        "--split", type=str, default="eval", choices=["train", "eval"],
        help="WildJailbreak config to load ('eval' = 2,210 held-out examples)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument(
        "--calibration_file", type=str, default=None,
        help="Path to calibration.json (applies temperature scaling and calibrated threshold)",
    )
    parser.add_argument(
        "--top_n", type=int, default=50,
        help="Max false negatives to print to console (0 = print all)",
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Optional CSV path to write all per-example predictions with error_type column",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers (mirrors eval_wildjailbreak.py)
# ---------------------------------------------------------------------------

def resolve_harmful_label_id(id2label: dict) -> int:
    for label_id, name in id2label.items():
        if name.lower() in ("nsfw", "harmful", "unsafe", "bad"):
            return int(label_id)
    return 1


def get_input_text(example: dict) -> str:
    if example["data_type"].startswith("adversarial"):
        return example["adversarial"]
    return example["vanilla"]


def get_true_label(example: dict) -> int:
    return 1 if "harmful" in example["data_type"] else 0


def batch_predict(model, tokenizer, texts, batch_size, max_seq_length, device, temperature=1.0):
    all_probs = []
    n = len(texts)
    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits / temperature, dim=-1).cpu().numpy()
        all_probs.extend(probs.tolist())
        done = min(start + batch_size, n)
        if (start // batch_size) % 5 == 0 or done == n:
            print(f"  [{done}/{n}] examples processed")
    return np.array(all_probs)


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print("="*70)


# ---------------------------------------------------------------------------
# Score-bucket histogram
# ---------------------------------------------------------------------------

def print_score_histogram(harmful_probs, true_labels, n_buckets=10):
    """Print an ASCII histogram of harmful_prob split by ground-truth label."""
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    labels_arr = np.array(true_labels)

    print_section("SCORE DISTRIBUTION (harmful_prob buckets)")
    header = f"  {'bucket':>14}  {'harmful (FN risk)':>18}  {'benign (FP risk)':>17}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        in_bucket = (harmful_probs >= lo) & (harmful_probs < hi)
        if i == n_buckets - 1:          # include right edge for last bucket
            in_bucket = (harmful_probs >= lo) & (harmful_probs <= hi)
        n_harm = int((in_bucket & (labels_arr == 1)).sum())
        n_ben  = int((in_bucket & (labels_arr == 0)).sum())
        bar_h = "#" * min(n_harm, 40)
        bar_b = "#" * min(n_ben,  40)
        bucket_label = f"[{lo:.1f}, {hi:.1f})"
        print(f"  {bucket_label:>14}  {n_harm:>5} {bar_h:<40}  {n_ben:>5} {bar_b}")


# ---------------------------------------------------------------------------
# False-negative table
# ---------------------------------------------------------------------------

def print_false_negatives(fn_rows, top_n: int):
    """
    Print false negatives sorted by harmful_prob ascending (worst misses first).

    fn_rows: list of dicts with keys: idx, data_type, text, harmful_prob
    """
    sorted_rows = sorted(fn_rows, key=lambda r: r["harmful_prob"])
    n_total = len(sorted_rows)
    display = sorted_rows if top_n == 0 else sorted_rows[:top_n]

    print_section(
        f"FALSE NEGATIVES — harmful prompts predicted as benign  "
        f"(showing {len(display)} of {n_total}, sorted by score asc)"
    )

    for rank, row in enumerate(display, 1):
        preview = row["text"].replace("\n", " ").strip()
        if len(preview) > TEXT_PREVIEW_CHARS:
            preview = preview[:TEXT_PREVIEW_CHARS] + "…"
        print(
            f"\n  [{rank:>3}] idx={row['idx']}  type={row['data_type']}"
            f"  score={row['harmful_prob']:.4f}"
        )
        print(f"       {preview}")


# ---------------------------------------------------------------------------
# Per-type summary
# ---------------------------------------------------------------------------

def print_per_type_summary(true_labels, pred_labels, harmful_probs, data_types):
    print_section("PER DATA_TYPE ERROR SUMMARY")
    true_arr = np.array(true_labels)
    pred_arr = np.array(pred_labels)

    for dt in sorted(set(data_types)):
        mask = np.array([d == dt for d in data_types])
        dt_true = true_arr[mask]
        dt_pred = pred_arr[mask]
        dt_probs = harmful_probs[mask]
        n = int(mask.sum())

        fn = int(((dt_true == 1) & (dt_pred == 0)).sum())  # false negatives
        fp = int(((dt_true == 0) & (dt_pred == 1)).sum())  # false positives
        correct = int((dt_true == dt_pred).sum())

        fn_scores = dt_probs[(dt_true == 1) & (dt_pred == 0)]
        fp_scores = dt_probs[(dt_true == 0) & (dt_pred == 1)]

        print(f"\n  [{dt}]  n={n}  correct={correct}  FN={fn}  FP={fp}")
        if len(fn_scores) > 0:
            print(
                f"    FN harmful_prob → "
                f"min={fn_scores.min():.4f}  "
                f"mean={fn_scores.mean():.4f}  "
                f"max={fn_scores.max():.4f}"
            )
        if len(fp_scores) > 0:
            print(
                f"    FP harmful_prob → "
                f"min={fp_scores.min():.4f}  "
                f"mean={fp_scores.mean():.4f}  "
                f"max={fp_scores.max():.4f}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    temperature = 1.0
    threshold = 0.5
    if args.calibration_file:
        with open(args.calibration_file) as f:
            calibration = json.load(f)
        temperature = calibration.get("temperature", 1.0)
        threshold = calibration.get("threshold", 0.5)
        print(f"Calibration: {args.calibration_file}")
        print(f"  temperature : {temperature:.4f}")
        print(f"  threshold   : {threshold:.4f}")

    # ------------------------------------------------------------------
    # Load model + tokenizer
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    print(f"Model  : {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    harmful_label_id = resolve_harmful_label_id(id2label)
    print(f"Labels : {id2label}")
    print(f"Harmful class → label ID {harmful_label_id} ('{id2label[harmful_label_id]}')")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print(f"\nLoading {EVAL_DATASET} (config='{args.split}') ...")
    raw = load_dataset(EVAL_DATASET, args.split, delimiter="\t", keep_default_na=False)
    ds = raw[list(raw.keys())[0]]
    print(f"Loaded {len(ds)} examples  |  columns: {ds.column_names}")

    dt_counts = Counter(ds["data_type"])
    for dt, cnt in sorted(dt_counts.items()):
        print(f"  {dt}: {cnt}")

    texts = [get_input_text(ex) for ex in ds]
    true_labels = [get_true_label(ex) for ex in ds]
    data_types = list(ds["data_type"])

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    print("\nRunning inference ...")
    probs = batch_predict(
        model, tokenizer, texts, args.batch_size, args.max_seq_length, device,
        temperature=temperature,
    )

    harmful_probs = probs[:, harmful_label_id]
    pred_labels = (harmful_probs >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Score histogram
    # ------------------------------------------------------------------
    print_score_histogram(harmful_probs, true_labels)

    # ------------------------------------------------------------------
    # Per-type error summary
    # ------------------------------------------------------------------
    print_per_type_summary(true_labels, pred_labels, harmful_probs, data_types)

    # ------------------------------------------------------------------
    # False negatives
    # ------------------------------------------------------------------
    true_arr = np.array(true_labels)
    fn_mask = (true_arr == 1) & (pred_labels == 0)
    fn_rows = [
        {
            "idx": i,
            "data_type": data_types[i],
            "text": texts[i],
            "harmful_prob": float(harmful_probs[i]),
        }
        for i in np.where(fn_mask)[0]
    ]

    print_false_negatives(fn_rows, top_n=args.top_n)

    # ------------------------------------------------------------------
    # Overall error counts
    # ------------------------------------------------------------------
    fp_mask = (true_arr == 0) & (pred_labels == 1)
    n_correct = int((true_arr == pred_labels).sum())
    print_section("SUMMARY")
    print(f"  Total examples    : {len(true_labels)}")
    print(f"  Correct           : {n_correct}")
    print(f"  False negatives   : {int(fn_mask.sum())}  (harmful → predicted benign)")
    print(f"  False positives   : {int(fp_mask.sum())}  (benign  → predicted harmful)")
    print(f"  Threshold used    : {threshold}")
    print(f"  Temperature used  : {temperature}")

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    if args.output_file:
        error_type = []
        for tl, pl in zip(true_labels, pred_labels.tolist()):
            if tl == 1 and pl == 0:
                error_type.append("false_negative")
            elif tl == 0 and pl == 1:
                error_type.append("false_positive")
            else:
                error_type.append("correct")

        df = pd.DataFrame({
            "data_type"   : data_types,
            "text"        : texts,
            "true_label"  : true_labels,
            "pred_label"  : pred_labels.tolist(),
            "harmful_prob": harmful_probs.tolist(),
            "error_type"  : error_type,
        })
        # Sort: false negatives first (ascending score), then false positives, then correct
        order = {"false_negative": 0, "false_positive": 1, "correct": 2}
        df["_sort_key"] = df["error_type"].map(order)
        df = df.sort_values(["_sort_key", "harmful_prob"]).drop(columns="_sort_key")

        out_dir = os.path.dirname(os.path.abspath(args.output_file))
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(args.output_file, index=False)
        print(f"\nAll predictions saved to: {args.output_file}")
        print(f"  Columns: {list(df.columns)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
