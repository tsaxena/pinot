"""
Evaluate a trained NSFW classifier on the held-out test split using a
previously computed calibration.json (temperature + thresholds).

Usage:
  python eval_test.py
  python eval_test.py --calibration_file ./my_output/calibration.json
  python eval_test.py --model_source tsaxena/distilbert-nsfw --calibration_file calibration.json
"""

import argparse
import json

from datasets import load_dataset, ClassLabel, Value
from scipy.special import softmax
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

DATASET_NAME = "eliasalbouzidi/NSFW-Safe-Dataset"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate with calibration on test split")
    parser.add_argument("--model_source", type=str, default="tsaxena/distilbert-nsfw",
                        help="Local path or HuggingFace Hub ID of the model")
    parser.add_argument("--calibration_file", type=str, default="calibration.json",
                        help="Path to calibration.json produced by calibrate.py")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    return parser.parse_args()


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
        for name, feat in features.items():
            if name != label_col and isinstance(feat, Value) and feat.dtype == "string":
                text_col = name
                break

    if text_col is None or label_col is None:
        raise ValueError(f"Could not detect text/label columns. Features: {features}")
    return text_col, label_col


def main():
    args = parse_args()

    # 1. Load calibration config
    with open(args.calibration_file) as f:
        cal = json.load(f)

    temperature = cal["temperature"]
    single_threshold = cal["threshold"]
    dual = cal.get("dual_threshold") or {}

    print(f"Calibration loaded from: {args.calibration_file}")
    print(f"  temperature    : {temperature:.4f}")
    print(f"  single threshold: {single_threshold:.4f}")
    if dual:
        print(f"  dual t_low     : {dual['t_low']:.4f}")
        print(f"  dual t_high    : {dual['t_high']:.4f}")

    # 2. Load the test split
    print(f"\nLoading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)
    text_col, label_col = find_text_and_label_columns(dataset)

    if "test" in dataset:
        test_split = dataset["test"]
    elif "validation" in dataset:
        test_split = dataset["validation"]
    else:
        test_split = dataset["train"].train_test_split(test_size=0.1, seed=42)["test"]

    print(f"Test examples: {len(test_split)}")

    # 3. Tokenise
    tokenizer = AutoTokenizer.from_pretrained(args.model_source)

    def tokenize(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=args.max_seq_length)

    cols_to_remove = [c for c in test_split.column_names if c not in ("labels",)]
    test_split = (
        test_split
        .map(lambda b: {"labels": b[label_col]}, batched=True)
        .map(tokenize, batched=True, remove_columns=cols_to_remove)
    )
    test_split.set_format("torch")

    # 4. Load model and collect raw logits
    model = AutoModelForSequenceClassification.from_pretrained(args.model_source)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="/tmp/eval_test_tmp",
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            report_to="none",
        ),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    print("\nRunning inference on test split...")
    pred_output = trainer.predict(test_split)
    logits = pred_output.predictions   # (N, num_labels)
    true_labels = pred_output.label_ids  # (N,)

    # 5. Apply temperature scaling → calibrated NSFW probabilities
    cal_probs = softmax(logits / temperature, axis=-1)[:, 1]

    # 6a. Single-threshold evaluation
    single_preds = (cal_probs >= single_threshold).astype(int)
    single_f1 = f1_score(true_labels, single_preds, zero_division=0)

    print("\n" + "=" * 60)
    print("SINGLE THRESHOLD RESULTS")
    print("=" * 60)
    print(f"Threshold : {single_threshold:.4f}")
    print(f"F1        : {single_f1:.4f}")
    print(classification_report(true_labels, single_preds, target_names=["safe", "nsfw"]))

    # 6b. Dual-threshold evaluation (if available)
    if dual:
        t_low = dual["t_low"]
        t_high = dual["t_high"]

        covered_mask = (cal_probs < t_low) | (cal_probs > t_high)
        abstain_mask = ~covered_mask
        coverage = covered_mask.mean()
        abstain_rate = abstain_mask.mean()

        covered_preds = (cal_probs[covered_mask] >= t_high).astype(int)
        covered_true = true_labels[covered_mask]
        dual_f1 = f1_score(covered_true, covered_preds, zero_division=0)

        print("\n" + "=" * 60)
        print("DUAL THRESHOLD RESULTS  (covered samples only)")
        print("=" * 60)
        print(f"t_low     : {t_low:.4f}")
        print(f"t_high    : {t_high:.4f}")
        print(f"Coverage  : {coverage:.2%}  ({covered_mask.sum()}/{len(cal_probs)} samples)")
        print(f"Abstain   : {abstain_rate:.2%}  ({abstain_mask.sum()} samples flagged for review)")
        print(f"F1 (covered): {dual_f1:.4f}")
        print(classification_report(covered_true, covered_preds, target_names=["safe", "nsfw"]))


if __name__ == "__main__":
    main()
