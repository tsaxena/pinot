# pinot

Fine-tune transformer models on the [eliasalbouzidi/NSFW-Safe-Dataset](https://huggingface.co/datasets/eliasalbouzidi/NSFW-Safe-Dataset) for binary text classification (NSFW vs. Safe).

## Setup

```bash
pip install -r requirements.txt
```

## Running

**Basic (with W&B logging):**
```bash
python train.py
```

**Disable W&B:**
```bash
python train.py --no_wandb
```

**Custom run:**
```bash
python train.py \
  --output_dir ./my-model \
  --num_train_epochs 5 \
  --per_device_train_batch_size 16 \
  --learning_rate 3e-5 \
  --weight_decay 0.01 \
  --max_seq_length 128 \
  --seed 42 \
  --wandb_project my-project \
  --wandb_run_name run-1
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `distilbert-base-uncased` | Model to fine-tune (`distilbert-base-uncased`, `distilroberta-base`, `roberta-base`) |
| `--output_dir` | `./<model_name>-nsfw` | Directory to save model checkpoints |
| `--num_train_epochs` | `3` | Number of training epochs |
| `--per_device_train_batch_size` | `32` | Training batch size per device |
| `--per_device_eval_batch_size` | `64` | Evaluation batch size per device |
| `--learning_rate` | `2e-5` | AdamW learning rate |
| `--weight_decay` | `0.01` | Weight decay |
| `--max_seq_length` | `128` | Max token length for truncation |
| `--seed` | `42` | Random seed |
| `--eval_strategy` | `epoch` | When to evaluate: `epoch` or `steps` |
| `--eval_steps` | `500` | Evaluate every N steps (only when `--eval_strategy=steps`) |
| `--wandb_project` | `distilbert-nsfw` | W&B project name |
| `--wandb_run_name` | auto | W&B run name |
| `--no_wandb` | `False` | Disable W&B logging |

## Evaluating on WildJailbreak

[allenai/wildjailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) is a held-out benchmark of 2,210 adversarial prompts (2,000 harmful + 210 benign). `eval_wildjailbreak.py` runs a trained checkpoint through the dataset and reports:

- **Overall**: accuracy, F1, precision, recall
- **Per data_type** (`vanilla_harmful`, `vanilla_benign`, `adversarial_harmful`, `adversarial_benign`):
  - accuracy
  - **Attack Success Rate (ASR)** — fraction of harmful prompts the model failed to flag
  - **Over-refusal Rate (ORR)** — fraction of benign prompts incorrectly flagged as harmful

Input text is selected per example: the `adversarial` column is used for adversarial data types; `vanilla` is used otherwise.

**Basic:**
```bash
python eval_wildjailbreak.py --model_path ./distilbert-base-uncased-nsfw
```

**Save per-example predictions:**
```bash
python eval_wildjailbreak.py --model_path ./distilbert-base-uncased-nsfw \
    --output_file predictions.csv
```

**Disable W&B:**
```bash
python eval_wildjailbreak.py --model_path ./distilbert-base-uncased-nsfw --no_wandb
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_path` | *(required)* | Local path or HuggingFace hub ID of the fine-tuned model |
| `--split` | `eval` | WildJailbreak config to load (`eval` = 2,210 held-out examples; `train` = 262K) |
| `--batch_size` | `64` | Inference batch size |
| `--max_seq_length` | `128` | Max token length for truncation |
| `--output_file` | `None` | Optional CSV path for per-example predictions |
| `--wandb_project` | `distilbert-nsfw` | W&B project name |
| `--wandb_run_name` | auto | W&B run name |
| `--no_wandb` | `False` | Disable W&B logging |

---

## Ablations

Run each model with identical hyperparameters to isolate the effect of architecture:

```bash
# Baseline
python train.py --model_name distilbert-base-uncased --wandb_run_name distilbert-baseline

# Ablation 1: DistilRoBERTa
python train.py --model_name distilroberta-base --wandb_run_name distilroberta-ablation

# Ablation 2: RoBERTa
python train.py --model_name roberta-base --wandb_run_name roberta-ablation
```

Each run saves its checkpoint to `./<model_name>-nsfw/` by default and logs to W&B under the same project for easy comparison.

## Baseline Hyperparameters

| Hyperparameter | Value |
|---|---|
| Model | `distilbert-base-uncased` |
| Epochs | `3` |
| Train batch size (per device) | `32` |
| Eval batch size (per device) | `64` |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Max sequence length | `128` |
| Optimizer | AdamW |
| LR scheduler | Linear (Transformers default) |
| NSFW class weight (`pos_weight`) | `1.66` |
| Best model metric | F1 |
| Seed | `42` |

## Loss Function

The model uses **weighted cross-entropy loss** to handle class imbalance between Safe and NSFW samples.

```
Loss = -sum_i [ w_i * y_i * log(p_i) ]
```

where class weights are `[1.0, 1.66]` for `[Safe, NSFW]`. The higher weight on the NSFW class (positive class) penalises the model more for missing NSFW content, improving recall on the minority class.

This is implemented in `WeightedTrainer.compute_loss` in `train.py`:

```python
weight = torch.tensor([1.0, POS_WEIGHT], device=outputs.logits.device)
loss = nn.CrossEntropyLoss(weight=weight)(outputs.logits, labels)
```

The best model checkpoint is selected by **F1 score** on the evaluation set.
