"""
Evaluate a fine-tuned safety classifier on a local prompt dataset.

Columns used
------------
  text         : prompt text (empty rows are skipped)
  safety_label : "Unsafe" → 1 (harmful), "Safe" → 0 (benign)
  source       : origin of the prompt (e.g. "4chan", "Lexica", "Template", "COCO")
  category     : optional — per-category metrics reported only if column is present

Metrics reported
----------------
  Overall         : accuracy, F1, precision, recall
  Per category    : accuracy
                    + attack_success_rate  (fraction of Unsafe examples predicted Safe)
                    + over_refusal_rate    (fraction of Safe examples predicted Unsafe)
                    (skipped if no category column)
  Per source      : same triplet as per-category

Usage
-----
  # Evaluate on the default local dataset
  python eval_unsafebench.py --model_path ./distilbert-base-uncased-nsfw

  # Specify a different CSV
  python eval_unsafebench.py --model_path ./distilbert-base-uncased-nsfw \\
      --data_file ./data/my_prompts.csv

  # Use calibration and save per-example predictions
  python eval_unsafebench.py --model_path tsaxena/distilbert-nsfw \\
      --calibration_file tsaxena-distilbert-nsfw-calibration/calibration.json \\
      --output_file preds.csv

  # Disable W&B
  python eval_unsafebench.py --model_path tsaxena/distilbert-nsfw --no_wandb
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import wandb
import evaluate
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    "data", "usafe_diffusion", "all_prompts.csv",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a safety classifier on UnsafeBench (text captions)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local path or HuggingFace hub ID of the fine-tuned model",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=DEFAULT_DATA_FILE,
        help="Path to the CSV file containing prompts (columns: text, safety_label, source[, category])",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Max tokenization length")
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional CSV path to write per-example predictions",
    )
    parser.add_argument(
        "--calibration_file",
        type=str,
        default=None,
        help="Path to calibration.json produced by calibrate.py "
             "(applies temperature scaling and calibrated threshold)",
    )
    parser.add_argument("--wandb_project", type=str, default="distilbert-nsfw")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_harmful_label_id(id2label: dict) -> int:
    """Return the label ID that the model uses for the harmful / NSFW class."""
    for label_id, name in id2label.items():
        if name.lower() in ("nsfw", "harmful", "unsafe", "bad"):
            return int(label_id)
    # Convention: label 1 is the positive (harmful) class
    return 1


def get_true_label(safety_label: str) -> int:
    """1 = Unsafe/harmful, 0 = Safe/benign."""
    return 1 if safety_label.strip().lower() == "unsafe" else 0


def batch_predict(model, tokenizer, texts, batch_size, max_seq_length, device, temperature=1.0):
    """
    Run batched inference.

    Parameters
    ----------
    temperature : float
        Temperature scaling factor from calibration.json.  Values > 1 soften the
        distribution; values < 1 sharpen it.  Use 1.0 (default) for no scaling.

    Returns
    -------
    probs : np.ndarray of shape (N, C) — temperature-scaled softmax probabilities
    """
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
    print(f"\n{'='*60}")
    print(title)
    print("="*60)


def eval_subset(true_labels, pred_labels, subset_name: str) -> dict:
    """
    Compute and print metrics for a single category or source bucket.

    Reports attack_success_rate for Unsafe examples (harmful that evaded
    detection) and over_refusal_rate for Safe examples (benign that were
    flagged).  When a bucket contains both labels, both rates are reported.
    """
    n = len(true_labels)
    acc = sum(t == p for t, p in zip(true_labels, pred_labels)) / n
    result = {"n": n, "accuracy": acc}

    n_unsafe = sum(true_labels)
    n_safe = n - n_unsafe

    if n_unsafe > 0:
        asr = sum(1 for t, p in zip(true_labels, pred_labels) if t == 1 and p == 0) / n_unsafe
        result["attack_success_rate"] = asr
        print(f"    accuracy           : {acc:.4f}")
        print(f"    attack_success_rate: {asr:.4f}  ({int(asr * n_unsafe)}/{n_unsafe} Unsafe examples predicted Safe)")
    if n_safe > 0:
        orr = sum(1 for t, p in zip(true_labels, pred_labels) if t == 0 and p == 1) / n_safe
        result["over_refusal_rate"] = orr
        if n_unsafe == 0:
            # Only safe examples in this bucket — print accuracy once
            print(f"    accuracy           : {acc:.4f}")
        print(f"    over_refusal_rate  : {orr:.4f}  ({int(orr * n_safe)}/{n_safe} Safe examples predicted Unsafe)")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    calibration = {}
    temperature = 1.0
    threshold = 0.5
    if args.calibration_file:
        with open(args.calibration_file) as f:
            calibration = json.load(f)
        temperature = calibration.get("temperature", 1.0)
        threshold = calibration.get("threshold", 0.5)
        print(f"Calibration loaded from: {args.calibration_file}")
        print(f"  temperature : {temperature:.4f}")
        print(f"  threshold   : {threshold:.4f}")
        if calibration.get("dual_threshold"):
            dt = calibration["dual_threshold"]
            print(f"  dual_threshold: t_low={dt['t_low']}, t_high={dt['t_high']}")

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    if args.no_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model_path": args.model_path,
                "data_file": args.data_file,
                "batch_size": args.batch_size,
                "max_seq_length": args.max_seq_length,
                "calibration_file": args.calibration_file,
                "calibration_temperature": temperature,
                "calibration_threshold": threshold,
            },
        )

    # ------------------------------------------------------------------
    # Load model + tokenizer
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
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
    print(f"\nLoading data from: {args.data_file}")
    df = pd.read_csv(args.data_file)
    print(f"Loaded {len(df)} rows  |  columns: {df.columns.tolist()}")

    label_counts = Counter(df["safety_label"])
    for lbl, cnt in sorted(label_counts.items()):
        print(f"  safety_label={lbl!r}: {cnt}")

    has_category = "category" in df.columns

    if has_category:
        category_counts = Counter(df["category"])
        print(f"\nCategories ({len(category_counts)}):")
        for cat, cnt in sorted(category_counts.items()):
            print(f"  {cat}: {cnt}")

    source_counts = Counter(df["source"])
    print(f"\nSources:")
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src}: {cnt}")

    # ------------------------------------------------------------------
    # Filter to examples with non-empty text
    # ------------------------------------------------------------------
    def has_text(row):
        t = row.get("text") if isinstance(row, dict) else row
        return isinstance(t, str) and t.strip() not in ("", "xxx", "N/A", "n/a")

    pre_filter = len(df)
    df = df[df["text"].apply(lambda t: isinstance(t, str) and t.strip() not in ("", "xxx", "N/A", "n/a"))].reset_index(drop=True)
    n_skipped = pre_filter - len(df)
    print(f"\nText filter: kept {len(df)}/{pre_filter} examples ({n_skipped} skipped — no text)")

    if len(df) == 0:
        print("ERROR: No examples with non-empty text found. Cannot evaluate a text classifier.")
        wandb.finish()
        return

    # ------------------------------------------------------------------
    # Prepare inputs
    # ------------------------------------------------------------------
    texts = df["text"].str.strip().tolist()
    true_labels = [get_true_label(v) for v in df["safety_label"]]
    categories = df["category"].tolist() if has_category else []
    sources = df["source"].tolist()

    n_unsafe = sum(true_labels)
    print(f"\nUnsafe: {n_unsafe}  |  Safe: {len(true_labels) - n_unsafe}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    print("\nRunning inference ...")
    probs = batch_predict(
        model, tokenizer, texts, args.batch_size, args.max_seq_length, device,
        temperature=temperature,
    )

    # Apply calibrated threshold on harmful-class probability; fall back to
    # argmax when no calibration file is provided (threshold stays at 0.5,
    # which is equivalent for a two-class model).
    harmful_probs = probs[:, harmful_label_id]
    pred_labels = (harmful_probs >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Overall metrics
    # ------------------------------------------------------------------
    true_list = [int(x) for x in true_labels]
    pred_list = pred_labels.tolist()

    acc_metric  = evaluate.load("accuracy")
    f1_metric   = evaluate.load("f1")
    prec_metric = evaluate.load("precision")
    rec_metric  = evaluate.load("recall")

    overall = {
        "accuracy" : acc_metric.compute(predictions=pred_list, references=true_list)["accuracy"],
        "f1"       : f1_metric.compute(predictions=pred_list, references=true_list, average="binary", pos_label=1)["f1"],
        "precision": prec_metric.compute(predictions=pred_list, references=true_list, average="binary", pos_label=1)["precision"],
        "recall"   : rec_metric.compute(predictions=pred_list, references=true_list, average="binary", pos_label=1)["recall"],
    }

    print_section("OVERALL METRICS")
    for k, v in overall.items():
        print(f"  {k:10s}: {v:.4f}")

    # ------------------------------------------------------------------
    # Per-category metrics (only when category column is present)
    # ------------------------------------------------------------------
    per_category_metrics = {}
    if has_category:
        print_section("PER CATEGORY METRICS")
        for cat in sorted(set(categories)):
            idxs = [i for i, c in enumerate(categories) if c == cat]
            cat_true = [true_labels[i] for i in idxs]
            cat_pred = pred_labels[idxs].tolist()
            print(f"\n  [{cat}]  n={len(idxs)}")
            per_category_metrics[cat] = eval_subset(cat_true, cat_pred, cat)

    # ------------------------------------------------------------------
    # Per-source metrics
    # ------------------------------------------------------------------
    print_section("PER SOURCE METRICS")
    per_source_metrics = {}
    for src in sorted(set(sources)):
        idxs = [i for i, s in enumerate(sources) if s == src]
        src_true = [true_labels[i] for i in idxs]
        src_pred = pred_labels[idxs].tolist()
        print(f"\n  [{src}]  n={len(idxs)}")
        per_source_metrics[src] = eval_subset(src_true, src_pred, src)

    # ------------------------------------------------------------------
    # Full classification report
    # ------------------------------------------------------------------
    print_section("CLASSIFICATION REPORT")
    print(classification_report(true_list, pred_list, target_names=["Safe", "Unsafe"], digits=4))

    # ------------------------------------------------------------------
    # Save per-example predictions (optional)
    # ------------------------------------------------------------------
    if args.output_file:
        df = pd.DataFrame({
            "category"    : categories,
            "source"      : sources,
            "text"        : texts,
            "true_label"  : true_labels,
            "pred_label"  : pred_labels,
            "harmful_prob": harmful_probs,
            "correct"     : (np.array(true_labels) == pred_labels).astype(int),
        })
        out_dir = os.path.dirname(os.path.abspath(args.output_file))
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(args.output_file, index=False)
        print(f"\nPredictions saved to: {args.output_file}")

    # ------------------------------------------------------------------
    # Dual-threshold evaluation (if calibration provides one)
    # ------------------------------------------------------------------
    dual = calibration.get("dual_threshold") if calibration else None

    # ------------------------------------------------------------------
    # W&B logging
    # ------------------------------------------------------------------
    log_dict = {f"eval/{k}": v for k, v in overall.items()}

    for cat, metrics in per_category_metrics.items():
        for k, v in metrics.items():
            if k != "n":
                log_dict[f"eval/category/{cat}/{k}"] = v

    for src, metrics in per_source_metrics.items():
        for k, v in metrics.items():
            if k != "n":
                log_dict[f"eval/source/{src}/{k}"] = v

    if dual and dual.get("t_low") is not None and dual.get("t_high") is not None:
        t_low, t_high = dual["t_low"], dual["t_high"]
        covered_mask = (harmful_probs < t_low) | (harmful_probs > t_high)
        n_covered = int(covered_mask.sum())
        n_abstained = len(harmful_probs) - n_covered
        dual_true = np.array(true_list)[covered_mask]
        dual_pred = (harmful_probs[covered_mask] >= t_high).astype(int)

        print_section("DUAL-THRESHOLD EVALUATION")
        print(f"  t_low={t_low}  t_high={t_high}")
        print(f"  Covered  : {n_covered} ({n_covered/len(harmful_probs):.2%})")
        print(f"  Abstained: {n_abstained} ({n_abstained/len(harmful_probs):.2%})")
        print(classification_report(dual_true, dual_pred, target_names=["Safe", "Unsafe"], digits=4))

        dual_overall = {
            "accuracy" : acc_metric.compute(predictions=dual_pred.tolist(), references=dual_true.tolist())["accuracy"],
            "f1"       : f1_metric.compute(predictions=dual_pred.tolist(), references=dual_true.tolist(), average="binary", pos_label=1)["f1"],
            "precision": prec_metric.compute(predictions=dual_pred.tolist(), references=dual_true.tolist(), average="binary", pos_label=1)["precision"],
            "recall"   : rec_metric.compute(predictions=dual_pred.tolist(), references=dual_true.tolist(), average="binary", pos_label=1)["recall"],
        }
        log_dict.update({f"eval/dual/{k}": v for k, v in dual_overall.items()})
        log_dict["eval/dual/coverage"] = n_covered / len(harmful_probs)
        log_dict["eval/dual/abstain_rate"] = n_abstained / len(harmful_probs)

    wandb.log(log_dict)
    wandb.finish()

    print(f"\nDone.  Evaluated on {len(texts)} examples from {args.data_file}.")
    return overall


if __name__ == "__main__":
    main()
