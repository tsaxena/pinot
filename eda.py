"""
Exploratory Data Analysis for eliasalbouzidi/NSFW-Safe-Dataset.
Outputs summary statistics to stdout and saves plots to assets/.
"""

import os
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset

DATASET_NAME = "eliasalbouzidi/NSFW-Safe-Dataset"
SPLITS = ["train", "validation", "test"]
LABEL_NAMES = {0: "Safe", 1: "NSFW"}
ASSETS_DIR = "assets"

os.makedirs(ASSETS_DIR, exist_ok=True)


def text_stats(texts):
    chars = [len(t) for t in texts]
    words = [len(t.split()) for t in texts]
    return {
        "count": len(texts),
        "char_mean": np.mean(chars),
        "char_median": np.median(chars),
        "char_std": np.std(chars),
        "char_min": np.min(chars),
        "char_max": np.max(chars),
        "word_mean": np.mean(words),
        "word_median": np.median(words),
        "word_std": np.std(words),
        "word_min": np.min(words),
        "word_max": np.max(words),
    }


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def main():
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    # ------------------------------------------------------------------
    # 1. Split sizes
    # ------------------------------------------------------------------
    print_section("Split Sizes")
    split_sizes = {}
    for split in SPLITS:
        n = len(dataset[split])
        split_sizes[split] = n
        print(f"  {split:<12}: {n:>7,} examples")
    total = sum(split_sizes.values())
    print(f"  {'TOTAL':<12}: {total:>7,} examples")

    # ------------------------------------------------------------------
    # 2. Label distribution
    # ------------------------------------------------------------------
    print_section("Label Distribution")
    dist = {}
    for split in SPLITS:
        labels = dataset[split]["labels"]
        counts = {v: labels.count(v) for v in sorted(set(labels))}
        dist[split] = counts
        parts = "  ".join(
            f"{LABEL_NAMES.get(k, k)}: {v:,} ({100*v/len(labels):.1f}%)"
            for k, v in counts.items()
        )
        print(f"  {split:<12}: {parts}")

    # ------------------------------------------------------------------
    # 3. Text length statistics
    # ------------------------------------------------------------------
    print_section("Text Length Statistics (train split)")
    train_texts = dataset["train"]["text"]
    stats = text_stats(train_texts)
    for label_id, label_name in LABEL_NAMES.items():
        subset = [t for t, l in zip(dataset["train"]["text"], dataset["train"]["labels"]) if l == label_id]
        s = text_stats(subset)
        print(f"\n  [{label_name}] n={s['count']:,}")
        print(f"    chars  — mean: {s['char_mean']:.0f}  median: {s['char_median']:.0f}  "
              f"std: {s['char_std']:.0f}  min: {s['char_min']}  max: {s['char_max']}")
        print(f"    words  — mean: {s['word_mean']:.1f}  median: {s['word_median']:.1f}  "
              f"std: {s['word_std']:.1f}  min: {s['word_min']}  max: {s['word_max']}")

    # ------------------------------------------------------------------
    # 4. Sample texts
    # ------------------------------------------------------------------
    print_section("Sample Texts (3 per class, train split)")
    for label_id, label_name in LABEL_NAMES.items():
        subset = [t for t, l in zip(dataset["train"]["text"], dataset["train"]["labels"]) if l == label_id]
        rng = np.random.default_rng(42)
        samples = rng.choice(subset, size=3, replace=False)
        print(f"\n  [{label_name}]")
        for i, s in enumerate(samples, 1):
            preview = textwrap.shorten(s, width=120, placeholder="...")
            print(f"    {i}. {preview}")

    # ------------------------------------------------------------------
    # 5. Plot: label distribution per split
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(SPLITS))
    width = 0.35
    safe_counts = [dist[s].get(0, 0) for s in SPLITS]
    nsfw_counts = [dist[s].get(1, 0) for s in SPLITS]
    bars1 = ax.bar(x - width / 2, safe_counts, width, label="Safe", color="#4c72b0")
    bars2 = ax.bar(x + width / 2, nsfw_counts, width, label="NSFW", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SPLITS])
    ax.set_ylabel("Example count")
    ax.set_title("Label Distribution per Split")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    for bar in [*bars1, *bars2]:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = os.path.join(ASSETS_DIR, "label_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved: {path}")

    # ------------------------------------------------------------------
    # 6. Plot: word-count distribution by class (train)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    colors = ["#4c72b0", "#dd8452"]
    for ax, (label_id, label_name), color in zip(axes, LABEL_NAMES.items(), colors):
        subset = [t for t, l in zip(dataset["train"]["text"], dataset["train"]["labels"]) if l == label_id]
        word_counts = [len(t.split()) for t in subset]
        ax.hist(word_counts, bins=60, color=color, edgecolor="white", linewidth=0.4)
        ax.set_title(f"Word Count Distribution — {label_name}")
        ax.set_xlabel("Word count")
        ax.set_ylabel("Frequency")
        ax.axvline(np.median(word_counts), color="black", linestyle="--", linewidth=1,
                   label=f"Median: {np.median(word_counts):.0f}")
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    plt.suptitle("Train Split — Word Count by Class", y=1.02)
    plt.tight_layout()
    path = os.path.join(ASSETS_DIR, "word_count_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
