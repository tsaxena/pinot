"""
Exploratory Data Analysis for yiting/UnsafeBench.

Since the model is a text classifier, a key focus is text-field coverage:
UnsafeBench is primarily an image benchmark and captions are optional.

Outputs summary statistics to stdout and saves plots to assets/unsafebench/.

Usage
-----
  python debug_unsafebench.py
"""

import os
import textwrap
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset

DATASET_NAME = "yiting/UnsafeBench"
SPLITS = ["train", "test"]
LABEL_NAMES = ["Safe", "Unsafe"]
ASSETS_DIR = os.path.join("assets", "unsafebench")
EMPTY_TEXT_SENTINELS = {"", "xxx", "n/a", "na", "none", "null"}

os.makedirs(ASSETS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)


def is_valid_text(t) -> bool:
    return isinstance(t, str) and t.strip().lower() not in EMPTY_TEXT_SENTINELS


def text_stats(texts: list[str]) -> dict:
    if not texts:
        return {}
    chars = [len(t) for t in texts]
    words = [len(t.split()) for t in texts]
    return {
        "count"       : len(texts),
        "char_mean"   : np.mean(chars),
        "char_median" : np.median(chars),
        "char_std"    : np.std(chars),
        "char_min"    : int(np.min(chars)),
        "char_max"    : int(np.max(chars)),
        "word_mean"   : np.mean(words),
        "word_median" : np.median(words),
        "word_std"    : np.std(words),
        "word_min"    : int(np.min(words)),
        "word_max"    : int(np.max(words)),
    }


def save(fig, filename: str):
    path = os.path.join(ASSETS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading dataset: {DATASET_NAME}")
    dataset = {}
    for split in SPLITS:
        dataset[split] = load_dataset(DATASET_NAME, split=split)
        # load_dataset can return DatasetDict or Dataset
        if hasattr(dataset[split], "keys"):
            dataset[split] = dataset[split][list(dataset[split].keys())[0]]

    # ------------------------------------------------------------------
    # 0. Column names
    # ------------------------------------------------------------------
    print_section("Column Names")
    for split in SPLITS:
        print(f"  {split:<8}: {dataset[split].column_names}")

    # ------------------------------------------------------------------
    # 1. Split sizes
    # ------------------------------------------------------------------
    print_section("Split Sizes")
    split_sizes = {}
    for split in SPLITS:
        n = len(dataset[split])
        split_sizes[split] = n
        print(f"  {split:<8}: {n:>6,} examples")
    print(f"  {'TOTAL':<8}: {sum(split_sizes.values()):>6,} examples")

    # ------------------------------------------------------------------
    # 2. Label distribution per split
    # ------------------------------------------------------------------
    print_section("Label Distribution (safety_label)")
    label_dist = {}
    for split in SPLITS:
        counts = Counter(dataset[split]["safety_label"])
        label_dist[split] = counts
        parts = "  ".join(
            f"{lbl}: {counts.get(lbl, 0):,} ({100 * counts.get(lbl, 0) / split_sizes[split]:.1f}%)"
            for lbl in LABEL_NAMES
        )
        print(f"  {split:<8}: {parts}")

    # ------------------------------------------------------------------
    # 3. Category distribution
    # ------------------------------------------------------------------
    print_section("Category Distribution (all splits combined)")
    all_categories = []
    all_labels_for_cat = []
    for split in SPLITS:
        all_categories.extend(dataset[split]["category"])
        all_labels_for_cat.extend(dataset[split]["safety_label"])

    cat_counts = Counter(all_categories)
    print(f"\n  {'Category':<30} {'Total':>7}  {'Safe':>7}  {'Unsafe':>7}  {'%Unsafe':>8}")
    print("  " + "-" * 62)
    for cat in sorted(cat_counts):
        idxs = [i for i, c in enumerate(all_categories) if c == cat]
        n_safe   = sum(1 for i in idxs if all_labels_for_cat[i] == "Safe")
        n_unsafe = sum(1 for i in idxs if all_labels_for_cat[i] == "Unsafe")
        total    = n_safe + n_unsafe
        pct      = 100 * n_unsafe / total if total else 0
        print(f"  {cat:<30} {total:>7,}  {n_safe:>7,}  {n_unsafe:>7,}  {pct:>7.1f}%")

    # ------------------------------------------------------------------
    # 4. Source distribution
    # ------------------------------------------------------------------
    print_section("Source Distribution")
    for split in SPLITS:
        counts = Counter(dataset[split]["source"])
        parts = "  ".join(
            f"{src}: {cnt:,} ({100 * cnt / split_sizes[split]:.1f}%)"
            for src, cnt in sorted(counts.items())
        )
        print(f"  {split:<8}: {parts}")

    # ------------------------------------------------------------------
    # 5. Text coverage
    # ------------------------------------------------------------------
    print_section("Text Coverage (non-empty captions)")
    text_coverage = {}
    for split in SPLITS:
        texts = dataset[split]["text"]
        n_valid = sum(1 for t in texts if is_valid_text(t))
        n_total = len(texts)
        text_coverage[split] = {"valid": n_valid, "total": n_total}
        print(f"  {split:<8}: {n_valid:,} / {n_total:,} have text ({100 * n_valid / n_total:.1f}%)")

    # Per-source text coverage (combined splits)
    print()
    all_sources = []
    all_texts = []
    all_labels_for_src = []
    for split in SPLITS:
        all_sources.extend(dataset[split]["source"])
        all_texts.extend(dataset[split]["text"])
        all_labels_for_src.extend(dataset[split]["safety_label"])

    src_counts = Counter(all_sources)
    for src in sorted(src_counts):
        idxs = [i for i, s in enumerate(all_sources) if s == src]
        n_valid = sum(1 for i in idxs if is_valid_text(all_texts[i]))
        n_total = len(idxs)
        print(f"  {src:<12}: {n_valid:,} / {n_total:,} have text ({100 * n_valid / n_total:.1f}%)")

    # Per-category text coverage
    print()
    for cat in sorted(cat_counts):
        idxs = [i for i, c in enumerate(all_categories) if c == cat]
        n_valid = sum(1 for i in idxs if is_valid_text(all_texts[i]))
        n_total = len(idxs)
        print(f"  {cat:<30}: {n_valid:,} / {n_total:,} ({100 * n_valid / n_total:.1f}%)")

    # ------------------------------------------------------------------
    # 6. Text length statistics (examples with valid text only)
    # ------------------------------------------------------------------
    print_section("Text Length Statistics (valid captions, train split)")
    train_texts_all = dataset["train"]["text"]
    train_labels_all = dataset["train"]["safety_label"]

    for lbl in LABEL_NAMES:
        subset = [
            t for t, l in zip(train_texts_all, train_labels_all)
            if l == lbl and is_valid_text(t)
        ]
        if not subset:
            print(f"\n  [{lbl}] — no valid text examples")
            continue
        s = text_stats(subset)
        print(f"\n  [{lbl}] n={s['count']:,}")
        print(f"    chars — mean: {s['char_mean']:.0f}  median: {s['char_median']:.0f}  "
              f"std: {s['char_std']:.0f}  min: {s['char_min']}  max: {s['char_max']}")
        print(f"    words — mean: {s['word_mean']:.1f}  median: {s['word_median']:.1f}  "
              f"std: {s['word_std']:.1f}  min: {s['word_min']}  max: {s['word_max']}")

    # ------------------------------------------------------------------
    # 7. Sample texts per label
    # ------------------------------------------------------------------
    print_section("Sample Captions (3 per label, train split)")
    rng = np.random.default_rng(42)
    for lbl in LABEL_NAMES:
        subset = [
            t for t, l in zip(train_texts_all, train_labels_all)
            if l == lbl and is_valid_text(t)
        ]
        if not subset:
            print(f"\n  [{lbl}] — no valid text examples")
            continue
        samples = rng.choice(subset, size=min(3, len(subset)), replace=False)
        print(f"\n  [{lbl}]")
        for i, s in enumerate(samples, 1):
            preview = textwrap.shorten(s.strip(), width=120, placeholder="...")
            print(f"    {i}. {preview}")

    # ------------------------------------------------------------------
    # Plot 1: Label distribution per split
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(SPLITS))
    width = 0.35
    safe_counts   = [label_dist[s].get("Safe",   0) for s in SPLITS]
    unsafe_counts = [label_dist[s].get("Unsafe", 0) for s in SPLITS]
    bars1 = ax.bar(x - width / 2, safe_counts,   width, label="Safe",   color="#4c72b0")
    bars2 = ax.bar(x + width / 2, unsafe_counts, width, label="Unsafe", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SPLITS])
    ax.set_ylabel("Example count")
    ax.set_title(f"Label Distribution per Split — {DATASET_NAME}")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    for bar in [*bars1, *bars2]:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    save(fig, "label_distribution.png")

    # ------------------------------------------------------------------
    # Plot 2: Category distribution, stacked Safe / Unsafe
    # ------------------------------------------------------------------
    cats_sorted = sorted(cat_counts)
    safe_per_cat   = []
    unsafe_per_cat = []
    for cat in cats_sorted:
        idxs = [i for i, c in enumerate(all_categories) if c == cat]
        safe_per_cat.append(  sum(1 for i in idxs if all_labels_for_cat[i] == "Safe"))
        unsafe_per_cat.append(sum(1 for i in idxs if all_labels_for_cat[i] == "Unsafe"))

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(cats_sorted))
    ax.bar(x, safe_per_cat,   label="Safe",   color="#4c72b0")
    ax.bar(x, unsafe_per_cat, bottom=safe_per_cat, label="Unsafe", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(cats_sorted, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Example count")
    ax.set_title(f"Category Distribution — {DATASET_NAME}")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    plt.tight_layout()
    save(fig, "category_distribution.png")

    # ------------------------------------------------------------------
    # Plot 3: Unsafe % per category (sorted)
    # ------------------------------------------------------------------
    pct_unsafe = [
        100 * u / (s + u) if (s + u) > 0 else 0
        for s, u in zip(safe_per_cat, unsafe_per_cat)
    ]
    order = np.argsort(pct_unsafe)[::-1]
    cats_ordered = [cats_sorted[i] for i in order]
    pcts_ordered = [pct_unsafe[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(np.arange(len(cats_ordered)), pcts_ordered, color="#dd8452")
    ax.set_xticks(np.arange(len(cats_ordered)))
    ax.set_xticklabels(cats_ordered, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("% Unsafe")
    ax.set_ylim(0, 105)
    ax.axhline(50, color="black", linestyle="--", linewidth=0.8, label="50%")
    ax.set_title(f"Fraction Unsafe per Category — {DATASET_NAME}")
    ax.legend()
    for bar, pct in zip(bars, pcts_ordered):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.0f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    save(fig, "unsafe_pct_per_category.png")

    # ------------------------------------------------------------------
    # Plot 4: Label distribution per source
    # ------------------------------------------------------------------
    srcs_sorted = sorted(src_counts)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(srcs_sorted))
    safe_per_src   = []
    unsafe_per_src = []
    for src in srcs_sorted:
        idxs = [i for i, s in enumerate(all_sources) if s == src]
        safe_per_src.append(  sum(1 for i in idxs if all_labels_for_src[i] == "Safe"))
        unsafe_per_src.append(sum(1 for i in idxs if all_labels_for_src[i] == "Unsafe"))
    bars1 = ax.bar(x - width / 2, safe_per_src,   width, label="Safe",   color="#4c72b0")
    bars2 = ax.bar(x + width / 2, unsafe_per_src, width, label="Unsafe", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(srcs_sorted)
    ax.set_ylabel("Example count")
    ax.set_title(f"Label Distribution per Source — {DATASET_NAME}")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    for bar in [*bars1, *bars2]:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    save(fig, "source_label_distribution.png")

    # ------------------------------------------------------------------
    # Plot 5: Text coverage per category
    # ------------------------------------------------------------------
    coverage_pct = []
    for cat in cats_sorted:
        idxs = [i for i, c in enumerate(all_categories) if c == cat]
        n_valid = sum(1 for i in idxs if is_valid_text(all_texts[i]))
        coverage_pct.append(100 * n_valid / len(idxs) if idxs else 0)

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(np.arange(len(cats_sorted)), coverage_pct, color="#55a868")
    ax.set_xticks(np.arange(len(cats_sorted)))
    ax.set_xticklabels(cats_sorted, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("% with non-empty text")
    ax.set_ylim(0, 105)
    ax.set_title(f"Text Caption Coverage per Category — {DATASET_NAME}")
    for bar, pct in zip(bars, coverage_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.0f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    save(fig, "text_coverage_per_category.png")

    # ------------------------------------------------------------------
    # Plot 6: Word-count distribution by label (valid captions, train)
    # ------------------------------------------------------------------
    label_texts = {
        lbl: [
            t for t, l in zip(train_texts_all, train_labels_all)
            if l == lbl and is_valid_text(t)
        ]
        for lbl in LABEL_NAMES
    }
    if any(label_texts.values()):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
        colors = ["#4c72b0", "#dd8452"]
        for ax, lbl, color in zip(axes, LABEL_NAMES, colors):
            subset = label_texts[lbl]
            if not subset:
                ax.set_title(f"{lbl} — no text data")
                continue
            word_counts = [len(t.split()) for t in subset]
            ax.hist(word_counts, bins=40, color=color, edgecolor="white", linewidth=0.4)
            ax.set_title(f"Word Count — {lbl}")
            ax.set_xlabel("Word count")
            ax.set_ylabel("Frequency")
            median_wc = np.median(word_counts)
            ax.axvline(median_wc, color="black", linestyle="--", linewidth=1,
                       label=f"Median: {median_wc:.0f}")
            ax.legend(fontsize=9)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
        plt.suptitle(f"Caption Word Count by Label — {DATASET_NAME} (train)", y=1.02)
        plt.tight_layout()
        save(fig, "word_count_distribution.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
