# pinot

Fine-tune DistilBERT on the [eliasalbouzidi/NSFW-Safe-Dataset](https://huggingface.co/datasets/eliasalbouzidi/NSFW-Safe-Dataset) for binary text classification (NSFW vs. Safe).

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
| `--output_dir` | `./distilbert-nsfw` | Directory to save model checkpoints |
| `--num_train_epochs` | `3` | Number of training epochs |
| `--per_device_train_batch_size` | `32` | Training batch size per device |
| `--per_device_eval_batch_size` | `64` | Evaluation batch size per device |
| `--learning_rate` | `2e-5` | AdamW learning rate |
| `--weight_decay` | `0.01` | Weight decay |
| `--max_seq_length` | `128` | Max token length for truncation |
| `--seed` | `42` | Random seed |
| `--wandb_project` | `distilbert-nsfw` | W&B project name |
| `--wandb_run_name` | auto | W&B run name |
| `--no_wandb` | `False` | Disable W&B logging |

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
