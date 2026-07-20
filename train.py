"""
Fine-tune a transformer model on eliasalbouzidi/NSFW-Safe-Dataset for text classification.

Supported models (pass via --model_name):
  distilbert-base-uncased                          (default, baseline)
  distilroberta-base                                (ablation)
  roberta-base                                      (ablation)
  distilbert-base-uncased-finetuned-sst-2-english   (ablation: starts from an SST-2
                                                      sentiment-finetuned checkpoint rather
                                                      than the raw pretrained base)
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import wandb
from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate
from sklearn.metrics import fbeta_score

DATASET_NAME = "eliasalbouzidi/NSFW-Safe-Dataset"
FBETA_BETA = 1.6  # weights recall (false negatives) over precision, per DiffGuard paper


class WeightedTrainer(Trainer):
    """Trainer that uses CrossEntropyLoss with data-driven class weights.

    Weights are computed as w_c = (1 - n_c / N) so that minority classes
    receive higher weight.
    """

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self.class_weights.to(outputs.logits.device)
        loss = nn.CrossEntropyLoss(weight=weight)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a transformer model for NSFW classification")
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        choices=[
            "distilbert-base-uncased",
            "distilroberta-base",
            "roberta-base",
            "distilbert-base-uncased-finetuned-sst-2-english",
        ],
        help="HuggingFace model to fine-tune",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save model (default: ./<model_name>-nsfw)")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of linear LR warmup steps (paper/model card uses 600)")
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        help="Optimizer passed to TrainingArguments (e.g. 'adamw_torch'). "
        "Note: this is always AdamW under the hood; to approximate the model "
        "card's plain Adam (no decoupled weight decay), pair this with --weight_decay 0.0.",
    )
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="fbeta",
        choices=["f1", "fbeta", "accuracy"],
        help="Metric used to select the best checkpoint. 'fbeta' (beta=1.6) matches the DiffGuard paper's checkpoint-selection criterion.",
    )
    parser.add_argument("--wandb_project", type=str, default="distilbert-nsfw", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (auto-generated if not set)")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="steps",
        choices=["epoch", "steps"],
        help="Evaluation strategy: 'steps' (default) or 'epoch'",
    )
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluate every N steps (only used when --eval_strategy=steps)")
    return parser.parse_args()


def find_text_and_label_columns(dataset):
    """Detect the text and label column names from dataset features."""
    features = dataset["train"].features

    # Identify label column
    label_col = None
    for name, feat in features.items():
        if isinstance(feat, ClassLabel):
            label_col = name
            break
    if label_col is None:
        # Fall back to columns named 'label' or 'labels'
        for candidate in ("label", "labels", "category", "class"):
            if candidate in features:
                label_col = candidate
                break

    # Identify text column
    text_col = None
    for candidate in ("text", "content", "sentence", "prompt", "message"):
        if candidate in features:
            text_col = candidate
            break
    if text_col is None:
        # Pick the first string column that isn't the label
        from datasets import Value
        for name, feat in features.items():
            if name != label_col and isinstance(feat, Value) and feat.dtype == "string":
                text_col = name
                break

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not detect text/label columns. Dataset features: {features}. "
            "Set text_col and label_col manually in the script."
        )

    print(f"Detected text column: '{text_col}', label column: '{label_col}'")
    return text_col, label_col


def get_label_info(dataset, label_col):
    """Return (num_labels, id2label, label2id)."""
    feat = dataset["train"].features[label_col]
    if isinstance(feat, ClassLabel):
        names = feat.names
        id2label = {i: n for i, n in enumerate(names)}
        label2id = {n: i for i, n in enumerate(names)}
        return len(names), id2label, label2id
    else:
        # Infer from unique values
        unique = sorted(set(dataset["train"][label_col]))
        id2label = {i: str(v) for i, v in enumerate(unique)}
        label2id = {str(v): i for i, v in enumerate(unique)}
        return len(unique), id2label, label2id


def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = f"./{args.model_name}-nsfw"

    # ------------------------------------------------------------------
    # 0. W&B
    # ------------------------------------------------------------------
    if args.no_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model": args.model_name,
                "dataset": DATASET_NAME,
                "num_train_epochs": args.num_train_epochs,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "optim": args.optim,
                "max_seq_length": args.max_seq_length,
                "seed": args.seed,
                "metric_for_best_model": args.metric_for_best_model,
                "fbeta_beta": FBETA_BETA,
            },
        )

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)
    print(dataset)
    for split, ds in dataset.items():
        print(f"  {split} columns: {ds.column_names}")

    text_col, label_col = find_text_and_label_columns(dataset)
    num_labels, id2label, label2id = get_label_info(dataset, label_col)
    print(f"Labels ({num_labels}): {id2label}")

    # ------------------------------------------------------------------
    # Compute class weights from training split: w_c = (1 - n_c / N)
    # ------------------------------------------------------------------
    raw_labels = dataset["train"][label_col]
    N = len(raw_labels)
    class_counts = [raw_labels.count(c) for c in range(num_labels)]
    class_weights = torch.tensor(
        [1.0 - n_c / N for n_c in class_counts], dtype=torch.float
    )
    print(f"Class counts : {dict(zip(id2label.values(), class_counts))}")
    print(f"Class weights: {dict(zip(id2label.values(), class_weights.tolist()))}")
    wandb.config.update({"class_weights": class_weights.tolist()})

    # Use 'test' split for evaluation if it exists, else split from train
    if "test" in dataset:
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
    elif "validation" in dataset:
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
    else:
        split = dataset["train"].train_test_split(test_size=0.1, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]

    # ------------------------------------------------------------------
    # 2. Tokenise
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=args.max_seq_length,
        )

    # Rename label column to 'labels' expected by Trainer
    def rename_label(batch):
        batch["labels"] = batch[label_col]
        return batch

    cols_to_remove = [c for c in train_dataset.column_names if c not in ("labels",)]

    train_dataset = (
        train_dataset
        .map(rename_label, batched=True)
        .map(tokenize, batched=True, remove_columns=cols_to_remove)
    )
    eval_dataset = (
        eval_dataset
        .map(rename_label, batched=True)
        .map(tokenize, batched=True, remove_columns=cols_to_remove)
    )

    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    # Note: distilbert-base-uncased-finetuned-sst-2-english already has a
    # 2-class classification head (positive/negative sentiment). Since
    # num_labels also equals 2 here, `from_pretrained` will NOT randomly
    # reinitialize that head by default -- it will silently reuse the
    # sentiment-trained weights as a warm start for the NSFW task, which
    # differs from the other base models (fresh/random head). This is a
    # deliberate ablation variable, not a bug, but worth knowing about
    # when comparing runs across --model_name values.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # ------------------------------------------------------------------
    # 4. Metrics
    # ------------------------------------------------------------------
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        avg = "binary" if num_labels == 2 else "macro"
        return {
            "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1_metric.compute(predictions=preds, references=labels, average=avg)["f1"],
            # Fbeta=1.6 weights recall over precision (0.3 FP / 0.7 FN), matching the
            # checkpoint-selection criterion used in the DiffGuard paper.
            "fbeta": fbeta_score(labels, preds, beta=FBETA_BETA, average=avg, zero_division=0),
        }

    # ------------------------------------------------------------------
    # 5. Train
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        optim=args.optim,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_strategy=args.eval_strategy,
        save_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=100,
        seed=args.seed,
        report_to="wandb",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    print("Starting training...")
    trainer.train()

    # ------------------------------------------------------------------
    # 6. Save best model
    # ------------------------------------------------------------------
    best_model_dir = os.path.join(args.output_dir, "best")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"Best model saved to {best_model_dir}")

    if trainer.state.best_metric is not None:
        print(f"Best eval/{args.metric_for_best_model}: {trainer.state.best_metric:.4f}")
        wandb.log({f"best_eval_{args.metric_for_best_model}": trainer.state.best_metric})

    wandb.finish()


if __name__ == "__main__":
    main()