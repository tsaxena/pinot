"""
Post-hoc calibration and threshold tuning for a trained NSFW classifier.

Steps:
  1. Load the best model from <model_dir>/best  OR  directly from --model_path
  2. Run inference on the eval split to collect logits + true labels
  3. Temperature scaling  – find T that minimises NLL on the eval set
  4. Single threshold     – sweep thresholds on calibrated probs, maximise F1
  5. Dual threshold       – find (t_low, t_high) pair; samples in the gap abstain
                           for human review; maximises F1 on covered samples
                           subject to coverage >= --min_coverage
  6. Save calibration.json to <model_dir>/best/  OR  to --output_dir

Usage (local training output):
  python calibrate.py --model_dir ./distilbert-base-uncased-nsfw
  python calibrate.py --model_dir ./roberta-base-nsfw --min_coverage 0.9

Usage (HuggingFace Hub model):
  python calibrate.py --model_path tsaxena/distilbert-nsfw
  python calibrate.py --model_path tsaxena/distilbert-nsfw --output_dir ./calibration_out
"""

import argparse
import json
import os

import numpy as np
import wandb
from datasets import load_dataset, ClassLabel
from scipy.optimize import minimize_scalar
from scipy.special import softmax as scipy_softmax
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless environments
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, precision_recall_curve, auc
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

DATASET_NAME = "eliasalbouzidi/NSFW-Safe-Dataset"


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate and tune threshold for a trained NSFW classifier")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model_dir", type=str, help="Root output dir used during training (contains best/ subfolder)")
    source.add_argument("--model_path", type=str, help="Direct path or HuggingFace Hub ID for the model checkpoint")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save calibration.json and auprc.png (only used with --model_path; defaults to ./<model-name>-calibration)")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="distilbert-nsfw")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--min_coverage",
        type=float,
        default=0.8,
        help="Minimum fraction of eval samples that must receive a definitive label in dual-threshold mode (default: 0.8)",
    )
    return parser.parse_args()


def tune_dual_threshold(cal_probs, true_labels, min_coverage=0.8):
    """Grid-search for (t_low, t_high) that maximises F1 on covered samples.

    A sample is *covered* when its calibrated NSFW probability is either
    below t_low (predict safe) or above t_high (predict NSFW).  Samples
    that fall in the gap [t_low, t_high] are flagged for human review.

    Returns a dict with keys: t_low, t_high, f1, coverage, abstain_rate.
    Returns None if no pair satisfies min_coverage.
    """
    candidates = np.linspace(0.05, 0.95, 91)  # 0.01 step – ~4k valid pairs
    best_f1 = -1.0
    best = None

    for t_low in candidates:
        for t_high in candidates:
            if t_high <= t_low:
                continue
            covered_mask = (cal_probs < t_low) | (cal_probs > t_high)
            coverage = covered_mask.mean()
            if coverage < min_coverage:
                continue
            preds = (cal_probs[covered_mask] >= t_high).astype(int)
            refs = true_labels[covered_mask]
            f1 = f1_score(refs, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best = {
                    "t_low": float(t_low),
                    "t_high": float(t_high),
                    "f1": float(f1),
                    "coverage": float(coverage),
                    "abstain_rate": float(1.0 - coverage),
                }

    return best


def find_text_and_label_columns(dataset):
    features = dataset["train"].features
    label_col = None
    for name, feat in features.items():
        if isinstance(feat, ClassLabel):
            label_col = name
            break
    if label_col is None:
        for candidate in ("label", "labels", "category", "class"):
            if candidate in features:
                label_col = candidate
                break

    text_col = None
    for candidate in ("text", "content", "sentence", "prompt", "message"):
        if candidate in features:
            text_col = candidate
            break
    if text_col is None:
        from datasets import Value
        for name, feat in features.items():
            if name != label_col and isinstance(feat, Value) and feat.dtype == "string":
                text_col = name
                break

    if text_col is None or label_col is None:
        raise ValueError(f"Could not detect text/label columns. Features: {features}")
    return text_col, label_col


def main():
    args = parse_args()

    # Resolve model source and output directory.
    if args.model_dir is not None:
        best_model_dir = os.path.join(args.model_dir, "best")
        if not os.path.isdir(best_model_dir):
            raise FileNotFoundError(
                f"Best model directory not found: {best_model_dir}\n"
                "Run train.py first or pass --model_dir pointing to the training output root."
            )
        model_source = best_model_dir
        output_dir = best_model_dir
        run_label = os.path.basename(args.model_dir)
    else:
        model_source = args.model_path
        run_label = args.model_path.replace("/", "-")
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = f"./{run_label}-calibration"
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 0. W&B
    # ------------------------------------------------------------------
    if args.no_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"calibrate-{run_label}",
        )

    # ------------------------------------------------------------------
    # 1. Load dataset (eval split only)
    # ------------------------------------------------------------------
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)
    text_col, label_col = find_text_and_label_columns(dataset)

    if "test" in dataset:
        eval_dataset = dataset["test"]
    elif "validation" in dataset:
        eval_dataset = dataset["validation"]
    else:
        eval_dataset = dataset["train"].train_test_split(test_size=0.1, seed=args.seed)["test"]

    # ------------------------------------------------------------------
    # 2. Tokenise
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_source)

    def tokenize(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=args.max_seq_length)

    cols_to_remove = [c for c in eval_dataset.column_names if c not in ("labels",)]

    eval_dataset = (
        eval_dataset
        .map(lambda b: {"labels": b[label_col]}, batched=True)
        .map(tokenize, batched=True, remove_columns=cols_to_remove)
    )
    eval_dataset.set_format("torch")

    # ------------------------------------------------------------------
    # 3. Load model and collect logits
    # ------------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(model_source)

    # Minimal TrainingArguments just for inference
    eval_args = TrainingArguments(
        output_dir="/tmp/calibrate_tmp",
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        seed=args.seed,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    print("Collecting logits on eval set...")
    pred_output = trainer.predict(eval_dataset)
    logits = pred_output.predictions   # (N, num_labels)
    true_labels = pred_output.label_ids  # (N,)

    # ------------------------------------------------------------------
    # 4. Temperature scaling: minimise NLL on eval set
    # ------------------------------------------------------------------
    def nll(temp):
        probs = scipy_softmax(logits / temp, axis=-1)
        correct_probs = probs[np.arange(len(true_labels)), true_labels]
        return -np.mean(np.log(correct_probs + 1e-12))

    opt = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    temperature = float(opt.x)
    print(f"Optimal temperature: {temperature:.4f}")

    # Calibrated probability for the positive (NSFW) class
    cal_probs = scipy_softmax(logits / temperature, axis=-1)[:, 1]

    # ------------------------------------------------------------------
    # 5. Threshold search: maximise F1 on eval set
    # ------------------------------------------------------------------
    thresholds = np.linspace(0.05, 0.95, 181)
    f1_scores = [
        f1_score(true_labels, (cal_probs >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    print(f"Optimal threshold: {best_threshold:.4f}  (calibrated eval F1 = {best_f1:.4f})")

    final_preds = (cal_probs >= best_threshold).astype(int)
    print("\nClassification report at optimal single threshold:")
    print(classification_report(true_labels, final_preds, target_names=["safe", "nsfw"]))

    # ------------------------------------------------------------------
    # 6. Dual threshold tuning
    # ------------------------------------------------------------------
    print(f"\nSearching for dual threshold (min_coverage={args.min_coverage:.0%})...")
    dual = tune_dual_threshold(cal_probs, true_labels, min_coverage=args.min_coverage)

    if dual is None:
        print(f"No (t_low, t_high) pair achieves coverage >= {args.min_coverage:.0%}. "
              "Try lowering --min_coverage.")
        dual_config = {}
    else:
        print(
            f"Dual threshold:  t_low={dual['t_low']:.4f}  t_high={dual['t_high']:.4f}\n"
            f"  Covered F1:    {dual['f1']:.4f}\n"
            f"  Coverage:      {dual['coverage']:.2%}\n"
            f"  Abstain rate:  {dual['abstain_rate']:.2%}"
        )
        dual_mask = (cal_probs < dual["t_low"]) | (cal_probs > dual["t_high"])
        dual_preds = (cal_probs[dual_mask] >= dual["t_high"]).astype(int)
        print("\nClassification report at dual threshold (covered samples only):")
        print(classification_report(true_labels[dual_mask], dual_preds, target_names=["safe", "nsfw"]))
        dual_config = dual

    # ------------------------------------------------------------------
    # 7. AUPRC plot
    # ------------------------------------------------------------------
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(true_labels, cal_probs)
    auprc = auc(recall_vals, precision_vals)
    print(f"\nAUPRC: {auprc:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall_vals, precision_vals, linewidth=2, label=f"PR curve (AUPRC = {auprc:.4f})")

    # Mark the optimal single threshold on the curve.
    # pr_thresholds has length N-1 relative to precision_vals/recall_vals,
    # so we find the index where the threshold crosses best_threshold.
    opt_idx = np.searchsorted(pr_thresholds, best_threshold, side="left")
    opt_idx = min(opt_idx, len(recall_vals) - 1)
    ax.scatter(
        recall_vals[opt_idx],
        precision_vals[opt_idx],
        marker="*",
        s=220,
        color="red",
        zorder=5,
        label=f"Single threshold = {best_threshold:.2f}  (F1={best_f1:.4f})",
    )

    # Mark dual thresholds if found.
    if dual_config:
        for label_name, t_val in [("t_low", dual_config["t_low"]), ("t_high", dual_config["t_high"])]:
            idx = np.searchsorted(pr_thresholds, t_val, side="left")
            idx = min(idx, len(recall_vals) - 1)
            ax.scatter(
                recall_vals[idx],
                precision_vals[idx],
                marker="D",
                s=90,
                zorder=5,
                label=f"Dual {label_name} = {t_val:.2f}",
            )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {run_label}")
    ax.legend(loc="lower left")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, "auprc.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"AUPRC plot saved to {plot_path}")

    # ------------------------------------------------------------------
    # 8. Log and save
    # ------------------------------------------------------------------
    log_payload = {
        "calibration_temperature": temperature,
        "optimal_threshold": best_threshold,
        "calibrated_threshold_f1": best_f1,
        "auprc": auprc,
    }
    if dual_config:
        log_payload.update({
            "dual_t_low": dual_config["t_low"],
            "dual_t_high": dual_config["t_high"],
            "dual_threshold_f1": dual_config["f1"],
            "dual_threshold_coverage": dual_config["coverage"],
            "dual_threshold_abstain_rate": dual_config["abstain_rate"],
        })
    wandb.log(log_payload)

    cal_config = {
        "temperature": temperature,
        "threshold": best_threshold,
        "calibrated_threshold_f1": best_f1,
        "auprc": auprc,
        "dual_threshold": dual_config,
    }
    cal_path = os.path.join(output_dir, "calibration.json")
    with open(cal_path, "w") as f:
        json.dump(cal_config, f, indent=2)
    print(f"\nCalibration config saved to {cal_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
