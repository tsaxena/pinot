"""
Evaluate a fine-tuned safety classifier on AIML-TUDA/i2p-benchmark.

I2P (Inappropriate Image Prompts) is an all-unsafe benchmark: every prompt was
used to generate inappropriate images, so the ground-truth label is always 1
(harmful).  The classifier's goal is to flag every prompt; the key metrics are:

  Detection Rate (DR) — fraction of prompts correctly flagged as harmful (recall)
  Attack Success Rate (ASR) — fraction of prompts that evaded detection (1 - DR)

Breakdowns reported
-------------------
  Overall         : accuracy, F1, precision, recall, ASR
  Per category    : ASR for each content category
                    (categories is multi-label; a prompt may appear in many buckets)
  Hard vs. easy   : ASR split on the `hard` flag

Usage
-----
  # Evaluate a local checkpoint
  python eval_i2p.py --model_path ./distilbert-base-uncased-nsfw

  # Use calibration and save per-example predictions
  python eval_i2p.py --model_path tsaxena/distilbert-nsfw \\
      --calibration_file tsaxena-distilbert-nsfw-calibration/calibration.json \\
      --output_file i2p_preds.csv

  # Disable W&B
  python eval_i2p.py --model_path tsaxena/distilbert-nsfw --no_wandb
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import wandb
import evaluate
from datasets import load_dataset
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

EVAL_DATASET = "AIML-TUDA/i2p-benchmark"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a safety classifier on I2P (Inappropriate Image Prompts)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local path or HuggingFace hub ID of the fine-tuned model",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load (I2P only has a 'train' split)",
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


def asr_for_subset(pred_labels, subset_name: str, n_total: int) -> float:
    """
    Compute and print ASR for a subset of harmful-only examples.

    All examples in I2P are harmful, so ASR is the only relevant rate:
    fraction predicted Safe (label 0) when the true label is Unsafe (label 1).
    """
    n = len(pred_labels)
    missed = sum(1 for p in pred_labels if p == 0)
    asr = missed / n if n > 0 else float("nan")
    dr = 1.0 - asr
    print(f"    n={n:<5d}  detection_rate={dr:.4f}  ASR={asr:.4f}  ({missed}/{n} evaded detection)")
    return asr


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
                "eval_dataset": EVAL_DATASET,
                "split": args.split,
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
    print(f"\nLoading {EVAL_DATASET} (split='{args.split}') ...")
    raw = load_dataset(EVAL_DATASET, split=args.split)
    # load_dataset may return a Dataset or a DatasetDict
    if hasattr(raw, "keys"):
        ds = raw[list(raw.keys())[0]]
    else:
        ds = raw
    print(f"Loaded {len(ds)} examples  |  columns: {ds.column_names}")

    # ------------------------------------------------------------------
    # Prepare inputs
    # All I2P examples are unsafe (label = 1).
    # ------------------------------------------------------------------
    texts = [ex["prompt"].strip() for ex in ds]
    true_labels = [1] * len(texts)   # all harmful by construction
    categories_per_example = [ex.get("categories") or [] for ex in ds]
    hard_flags = [bool(ex.get("hard", False)) for ex in ds]

    # Summarise category distribution (multi-label)
    cat_counts: dict = defaultdict(int)
    for cats in categories_per_example:
        for c in cats:
            cat_counts[c] += 1
    print(f"\nContent categories ({len(cat_counts)} unique):")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}")

    n_hard = sum(hard_flags)
    print(f"\nHard examples: {n_hard}  |  Easy examples: {len(texts) - n_hard}")

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
    # Overall metrics
    # I2P is all-harmful so precision/recall map cleanly to detection rate.
    # ------------------------------------------------------------------
    true_list = [int(x) for x in true_labels]
    pred_list = pred_labels.tolist()

    acc_metric  = evaluate.load("accuracy")
    f1_metric   = evaluate.load("f1")
    prec_metric = evaluate.load("precision")
    rec_metric  = evaluate.load("recall")

    overall = {
        "accuracy"       : acc_metric.compute(predictions=pred_list, references=true_list)["accuracy"],
        "f1"             : f1_metric.compute(predictions=pred_list, references=true_list, average="binary", pos_label=1)["f1"],
        "precision"      : prec_metric.compute(predictions=pred_list, references=true_list, average="binary", pos_label=1)["precision"],
        "recall"         : rec_metric.compute(predictions=pred_list, references=true_list, average="binary", pos_label=1)["recall"],
        "detection_rate" : float(np.mean(pred_labels == 1)),
        "attack_success_rate": float(np.mean(pred_labels == 0)),
    }

    print_section("OVERALL METRICS")
    for k, v in overall.items():
        print(f"  {k:20s}: {v:.4f}")

    # ------------------------------------------------------------------
    # Per-category metrics (multi-label: each example counted once per category)
    # ------------------------------------------------------------------
    print_section("PER CATEGORY METRICS")
    per_category_asr = {}
    all_cats = sorted(cat_counts.keys())
    for cat in all_cats:
        idxs = [i for i, cats in enumerate(categories_per_example) if cat in cats]
        if not idxs:
            continue
        cat_pred = pred_labels[idxs]
        print(f"\n  [{cat}]")
        per_category_asr[cat] = asr_for_subset(cat_pred, cat, len(texts))

    # ------------------------------------------------------------------
    # Hard vs. easy breakdown
    # ------------------------------------------------------------------
    print_section("HARD vs. EASY BREAKDOWN")
    hard_idxs = [i for i, h in enumerate(hard_flags) if h]
    easy_idxs = [i for i, h in enumerate(hard_flags) if not h]

    if hard_idxs:
        print(f"\n  [hard]")
        hard_asr = asr_for_subset(pred_labels[hard_idxs], "hard", len(texts))
    else:
        hard_asr = float("nan")
        print("  [hard]  n=0 — no hard examples in this split")

    if easy_idxs:
        print(f"\n  [easy]")
        easy_asr = asr_for_subset(pred_labels[easy_idxs], "easy", len(texts))
    else:
        easy_asr = float("nan")
        print("  [easy]  n=0 — no easy examples in this split")

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
            "prompt"         : texts,
            "categories"     : [", ".join(c) for c in categories_per_example],
            "hard"           : hard_flags,
            "true_label"     : true_labels,
            "pred_label"     : pred_labels,
            "harmful_prob"   : harmful_probs,
            "correct"        : (pred_labels == np.array(true_labels)).astype(int),
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

    for cat, asr in per_category_asr.items():
        log_dict[f"eval/category/{cat}/attack_success_rate"] = asr
        log_dict[f"eval/category/{cat}/detection_rate"] = 1.0 - asr

    if not np.isnan(hard_asr):
        log_dict["eval/hard/attack_success_rate"] = hard_asr
        log_dict["eval/hard/detection_rate"] = 1.0 - hard_asr
    if not np.isnan(easy_asr):
        log_dict["eval/easy/attack_success_rate"] = easy_asr
        log_dict["eval/easy/detection_rate"] = 1.0 - easy_asr

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

    print(f"\nDone.  Evaluated on {len(texts)} I2P prompts.")
    return overall


if __name__ == "__main__":
    main()
