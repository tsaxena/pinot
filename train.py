"""
Quick sanity check: confirm which integer id corresponds to 'nsfw' vs 'safe'
in eliasalbouzidi/NSFW-Safe-Dataset, before relying on fbeta's average='binary'
(which assumes label 1 is the positive class).

Run: python check_label_mapping.py
"""

from datasets import load_dataset, ClassLabel

DATASET_NAME = "eliasalbouzidi/NSFW-Safe-Dataset"

dataset = load_dataset(DATASET_NAME)
features = dataset["train"].features
print("All features:", features)

# Find the label column (same logic as train.py's find_text_and_label_columns)
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

print(f"\nLabel column: '{label_col}'")

feat = features[label_col]
if isinstance(feat, ClassLabel):
    id2label = {i: n for i, n in enumerate(feat.names)}
    label2id = {n: i for i, n in enumerate(feat.names)}
else:
    unique = sorted(set(dataset["train"][label_col]))
    id2label = {i: str(v) for i, v in enumerate(unique)}
    label2id = {str(v): i for i, v in enumerate(unique)}

print(f"id2label: {id2label}")
print(f"label2id: {label2id}")

# The check that actually matters for fbeta_score(..., average='binary'):
# sklearn's 'binary' mode treats label value 1 as the positive class by default.
nsfw_id = label2id.get("nsfw") or label2id.get("NSFW") or label2id.get("unsafe")
print(f"\n'nsfw' maps to id: {nsfw_id}")
if nsfw_id == 1:
    print("OK: nsfw == 1, matches sklearn's default positive class for average='binary'.")
elif nsfw_id == 0:
    print(
        "WARNING: nsfw == 0, but average='binary' in fbeta_score/f1_metric defaults to "
        "treating id 1 as positive. Fbeta/F1 will be computed for the WRONG class "
        "(i.e. optimizing for 'safe' detection, not 'nsfw' detection) unless you pass "
        "pos_label=0 explicitly to fbeta_score (and check the f1 evaluate.load metric too)."
    )
else:
    print("Could not automatically detect the nsfw label id — inspect id2label above manually.")

# Cross-check against a few raw examples
print("\nSample rows (first 5 train examples):")
for i in range(5):
    row = dataset["train"][i]
    print(f"  label={row[label_col]!r}  ({id2label.get(row[label_col], '?')})  text={str(row.get('text', ''))[:60]!r}")