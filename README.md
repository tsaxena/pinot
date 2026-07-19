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
| `--eval_strategy` | `steps` | When to evaluate: `steps` or `epoch` |
| `--eval_steps` | `100` | Evaluate every N steps (only when `--eval_strategy=steps`) |
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

## Calibration and Threshold Tuning

`calibrate.py` runs post-hoc on a trained checkpoint and produces a `calibration.json` file that should be loaded at inference time.

**Steps performed:**

1. **Temperature scaling** — finds a scalar temperature `T` that minimises negative log-likelihood on the eval set. Applies `softmax(logits / T)` to produce better-calibrated probabilities.
2. **Single threshold** — sweeps 181 thresholds (0.05 → 0.95) on the calibrated NSFW-class probability and picks the one that maximises F1.
3. **Dual threshold** — searches over all `(t_low, t_high)` pairs. Samples with prob `< t_low` are labelled safe; samples with prob `> t_high` are labelled NSFW; samples in the gap `[t_low, t_high]` abstain for human review. The pair that maximises F1 on covered samples subject to `coverage >= --min_coverage` is selected.

**Basic:**
```bash
python calibrate.py --model_dir ./distilbert-base-uncased-nsfw
```

**Require 90% of samples to receive a definitive label:**
```bash
python calibrate.py --model_dir ./distilbert-base-uncased-nsfw --min_coverage 0.9
```

**Disable W&B:**
```bash
python calibrate.py --model_dir ./distilbert-base-uncased-nsfw --no_wandb
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_dir` | *(required)* | Root output dir from training (must contain a `best/` subfolder) |
| `--min_coverage` | `0.8` | Minimum fraction of eval samples that must receive a definitive label in dual-threshold mode |
| `--max_seq_length` | `128` | Max token length for truncation |
| `--per_device_eval_batch_size` | `64` | Evaluation batch size per device |
| `--seed` | `42` | Random seed |
| `--wandb_project` | `distilbert-nsfw` | W&B project name |
| `--wandb_run_name` | auto | W&B run name |
| `--no_wandb` | `False` | Disable W&B logging |

### Output

Saves `<model_dir>/best/calibration.json`:

```json
{
  "temperature": 1.23,
  "threshold": 0.42,
  "calibrated_threshold_f1": 0.95,
  "dual_threshold": {
    "t_low": 0.25,
    "t_high": 0.70,
    "f1": 0.97,
    "coverage": 0.83,
    "abstain_rate": 0.17
  }
}
```

---

## Calibrated Evaluation on Test Split

`eval_test.py` runs the held-out test split of `eliasalbouzidi/NSFW-Safe-Dataset` through the model using a `calibration.json` produced by `calibrate.py`. It applies temperature scaling and the tuned threshold(s) and reports a classification report for both single- and dual-threshold modes.

**Basic (uses `calibration.json` in the current directory):**
```bash
python eval_test.py
```

**Specify model and calibration file explicitly:**
```bash
python eval_test.py \
  --model_source ./distilbert-base-uncased-nsfw/best \
  --calibration_file ./distilbert-base-uncased-nsfw/best/calibration.json
```

**HuggingFace Hub model with a local calibration file:**
```bash
python eval_test.py \
  --model_source tsaxena/distilbert-nsfw \
  --calibration_file ./calibration_out/calibration.json
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_source` | `tsaxena/distilbert-nsfw` | Local path or HuggingFace Hub ID of the model |
| `--calibration_file` | `calibration.json` | Path to `calibration.json` produced by `calibrate.py` |
| `--max_seq_length` | `128` | Max token length for truncation |
| `--per_device_eval_batch_size` | `64` | Evaluation batch size per device |

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

### DistilRoBERTa

**Train:**
```bash
python train.py --model_name distilroberta-base --wandb_run_name distilroberta-ablation
```

**Calibrate:**
```bash
python calibrate.py --model_dir ./distilroberta-base-nsfw
```

**Evaluate on WildJailbreak:**
```bash
python eval_wildjailbreak.py --model_path ./distilroberta-base-nsfw/best
```

### RoBERTa

**Train:**
```bash
python train.py --model_name roberta-base --wandb_run_name roberta-ablation
```

**Calibrate:**
```bash
python calibrate.py --model_dir ./roberta-base-nsfw
```

**Evaluate on WildJailbreak:**
```bash
python eval_wildjailbreak.py --model_path ./roberta-base-nsfw/best
```

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

---

## Uploading to HuggingFace Hub

After training (and optionally calibrating), push the best checkpoint to the Hub with `upload.py`. It uploads the model, tokenizer, and any extra artefacts (`calibration.json`, `auprc.png`) found in `<model_dir>/best/`.

**Authenticate first:**
```bash
huggingface-cli login
# or export HF_TOKEN=<your-token>
```

**Basic:**
```bash
python upload.py \
  --model_dir ./distilbert-base-uncased-nsfw \
  --hub_model_id username/distilbert-nsfw
```

**Private repository:**
```bash
python upload.py \
  --model_dir ./distilbert-base-uncased-nsfw \
  --hub_model_id username/distilbert-nsfw \
  --private
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_dir` | *(required)* | Root output dir from training (must contain a `best/` subfolder) |
| `--hub_model_id` | *(required)* | HuggingFace Hub repo ID, e.g. `username/distilbert-nsfw` |
| `--commit_message` | `Upload best checkpoint` | Commit message for the Hub push |
| `--private` | `False` | Create the Hub repository as private |
