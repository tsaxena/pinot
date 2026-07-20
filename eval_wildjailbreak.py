"""
Evaluate a fine-tuned safety classifier on allenai/wildjailbreak.

The model is expected to distinguish harmful / NSFW prompts from safe / benign ones.
Input text is selected per example:
  - adversarial_* data types  →  'adversarial' column (the jailbreak-crafted version)
  - vanilla_* data types      →  'vanilla' column (the direct prompt)
Ground-truth labels: *_harmful → 1 (harmful), *_benign → 0 (benign).

Metrics reported
----------------
  Overall  : accuracy, F1, precision, recall
  Per type : accuracy
             + attack_success_rate  (fraction of harmful prompts that evaded detection)
             + over_refusal_rate    (fraction of benign prompts incorrectly flagged)

Usage
-----
  # Evaluate a local checkpoint on the eval split (default)
  python eval_wildjailbreak.py --model_path ./distilbert-base-uncased-nsfw

  # Evaluate on the training split, save per-example predictions
  python eval_wildjailbreak.py --model_path ./distilbert-base-uncased-nsfw \\
      --split train --output_file predictions.csv

  # Disable W&B
  python eval_wildjailbreak.py --model_path ./distilbert-base-uncased-nsfw --no_wandb
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
from datasets import load_dataset
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

EVAL_DATASET = "allenai/wildjailbreak"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a safety classifier on WildJailbreak",
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
        default="eval",
        choices=["train", "eval"],
        help="WildJailbreak config to load ('eval' = 2,210 held-out examples)",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Max tokenization length")
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional CSV path to write per-example predictions",
    )
    parser.add_argument("--calibration_file", type=str, default=None,
        help="Path to calibration.json produced by calibrate.py (applies temperature scaling and calibrated threshold)")
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


def get_input_text(example: dict) -> str:
    """Pick the right column based on data_type."""
    if example["data_type"].startswith("adversarial"):
        return example["adversarial"]
    return example["vanilla"]


def get_true_label(example: dict) -> int:
    """1 = harmful, 0 = benign."""
    return 1 if "harmful" in example["data_type"] else 0


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
    raw_preds : np.ndarray of shape (N,)   — argmax label IDs from the model
    probs     : np.ndarray of shape (N, C) — temperature-scaled softmax probabilities
    """
    all_preds, all_probs = [], []
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
        all_preds.extend(np.argmax(probs, axis=-1).tolist())
        all_probs.extend(probs.tolist())
        done = min(start + batch_size, n)
        if (start // batch_size) % 5 == 0 or done == n:
            print(f"  [{done}/{n}] examples processed")
    return np.array(all_preds), np.array(all_probs)


def print_section(title: str):
    print(f"\n{'='*60}")
    print(title)
    print("="*60)


def eval_subset(true_labels, pred_labels, data_type_name: str) -> dict:
    """
    Compute and print metrics for a single data_type bucket.

    For harmful subsets  → reports attack_success_rate (ASR).
    For benign subsets   → reports over_refusal_rate (ORR).
    """
    n = len(true_labels)
    acc = sum(t == p for t, p in zip(true_labels, pred_labels)) / n
    result = {"n": n, "accuracy": acc}

    if "harmful" in data_type_name:
        asr = sum(1 for p in pred_labels if p == 0) / n
        result["attack_success_rate"] = asr
        print(f"    accuracy           : {acc:.4f}")
        print(f"    attack_success_rate: {asr:.4f}  ({int(asr * n)}/{n} harmful prompts evaded detection)")
    else:
        orr = sum(1 for p in pred_labels if p == 1) / n
        result["over_refusal_rate"] = orr
        print(f"    accuracy           : {acc:.4f}")
        print(f"    over_refusal_rate  : {orr:.4f}  ({int(orr * n)}/{n} benign prompts flagged as harmful)")

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
    print(f"\nLoading {EVAL_DATASET} (config='{args.split}') ...")
    raw = load_dataset(EVAL_DATASET, args.split, delimiter="\t", keep_default_na=False)
    # WildJailbreak loads configs as DatasetDict with a single 'train' split key
    ds = raw[list(raw.keys())[0]]
    print(f"Loaded {len(ds)} examples  |  columns: {ds.column_names}")

    dt_counts = Counter(ds["data_type"])
    for dt, cnt in sorted(dt_counts.items()):
        print(f"  {dt}: {cnt}")

    # ------------------------------------------------------------------
    # Prepare inputs
    # ------------------------------------------------------------------
    texts = [get_input_text(ex) for ex in ds]
    true_labels = [get_true_label(ex) for ex in ds]
    data_types = ds["data_type"]

    n_harmful = sum(true_labels)
    print(f"\nHarmful: {n_harmful}  |  Benign: {len(true_labels) - n_harmful}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    print("\nRunning inference ...")
    raw_preds, probs = batch_predict(
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
    # Per data_type metrics
    # ------------------------------------------------------------------
    print_section("PER DATA_TYPE METRICS")
    per_type_metrics = {}
    for dt in sorted(set(data_types)):
        idxs = [i for i, d in enumerate(data_types) if d == dt]
        dt_true = [true_labels[i] for i in idxs]
        dt_pred = pred_labels[idxs].tolist()
        print(f"\n  [{dt}]  n={len(idxs)}")
        per_type_metrics[dt] = eval_subset(dt_true, dt_pred, dt)

    # ------------------------------------------------------------------
    # Full classification report
    # ------------------------------------------------------------------
    print_section("CLASSIFICATION REPORT")
    print(classification_report(true_list, pred_list, target_names=["benign", "harmful"], digits=4))

    # ------------------------------------------------------------------
    # Save per-example predictions (optional)
    # ------------------------------------------------------------------
    if args.output_file:
        df = pd.DataFrame({
            "data_type"   : data_types,
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
    for dt, metrics in per_type_metrics.items():
        for k, v in metrics.items():
            if k != "n":
                log_dict[f"eval/{dt}/{k}"] = v

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
        print(classification_report(dual_true, dual_pred, target_names=["benign", "harmful"], digits=4))

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

    print("\nDone.")
    return overall


if __name__ == "__main__":
    main()
