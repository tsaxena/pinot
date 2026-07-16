"""
Upload a trained best checkpoint to the HuggingFace Hub.

Loads the model and tokenizer from <model_dir>/best/ and pushes them to the
specified Hub repository. If calibration.json or auprc.png are present they
are also uploaded.

Requires HuggingFace authentication:
  huggingface-cli login
  # or set the HF_TOKEN environment variable

Usage:
  python upload.py --model_dir ./distilbert-base-uncased-nsfw --hub_model_id username/distilbert-nsfw
  python upload.py --model_dir ./distilroberta-base-nsfw     --hub_model_id username/distilroberta-nsfw
  python upload.py --model_dir ./roberta-base-nsfw           --hub_model_id username/roberta-nsfw
  python upload.py --model_dir ./distilbert-base-uncased-nsfw --hub_model_id username/distilbert-nsfw --private
"""

import argparse
import os

from huggingface_hub import HfApi
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Upload best model checkpoint to HuggingFace Hub")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Root output dir from training (must contain a best/ subfolder)",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        required=True,
        help="HuggingFace Hub repo ID, e.g. 'username/distilbert-nsfw'",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload best checkpoint",
        help="Commit message for the Hub push (default: 'Upload best checkpoint')",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hub repository as private",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    best_model_dir = os.path.join(args.model_dir, "best")

    if not os.path.isdir(best_model_dir):
        raise FileNotFoundError(
            f"Best model directory not found: {best_model_dir}\n"
            "Run train.py first or pass --model_dir pointing to the training output root."
        )

    # ------------------------------------------------------------------
    # 1. Push model and tokenizer
    # ------------------------------------------------------------------
    print(f"Loading model and tokenizer from {best_model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)

    print(f"Pushing model to {args.hub_model_id} ...")
    model.push_to_hub(
        args.hub_model_id,
        commit_message=args.commit_message,
        private=args.private,
    )

    print("Pushing tokenizer ...")
    tokenizer.push_to_hub(
        args.hub_model_id,
        commit_message=args.commit_message,
        private=args.private,
    )

    # ------------------------------------------------------------------
    # 2. Upload extra artefacts (calibration config, AUPRC plot)
    # ------------------------------------------------------------------
    api = HfApi()

    extra_files = {
        "calibration.json": os.path.join(best_model_dir, "calibration.json"),
        "auprc.png": os.path.join(best_model_dir, "auprc.png"),
    }

    for repo_filename, local_path in extra_files.items():
        if os.path.isfile(local_path):
            print(f"Uploading {repo_filename} ...")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_filename,
                repo_id=args.hub_model_id,
                commit_message=f"{args.commit_message}: add {repo_filename}",
            )
        else:
            print(f"Skipping {repo_filename} (not found in {best_model_dir})")

    print(f"\nDone. Model available at https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
