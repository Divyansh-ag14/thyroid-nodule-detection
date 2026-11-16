"""
Dataset indexing, leakage-safe splitting and caching.

Two sources:
- TN5000 (Zhang et al., Scientific Data 2025; CC BY): 5,000 biopsy-confirmed B-mode
  images with one nodule bounding box each and an official image-level 7:1:2 split.
  Primary development + test data.
- Kaggle "thyroid-cancer-classification-ultrasound-dataset": 3,115 images of unknown
  provenance. Used as an EXTERNAL test set for TN5000 models (never trained on), and as
  the development set for the training-recipe sweep (`--dataset kaggle`).

Neither source has patient IDs, and image-level splits leak near-identical frames of
the same sweep. For each dataset we:

1. Compute two independent perceptual hashes per image: a 256-bit difference
   hash (gradient structure) and a 255-bit DCT hash (low-frequency content).
2. Union images within `GROUP_DIST` dHash or `PHASH_DIST` pHash bits into scan groups
   (a proxy for "same scan / same patient").
3. Drop near-identical images (<= `DUP_DIST` dHash bits) that carry contradicting labels.
4. Assign splits so no scan group spans two splits. TN5000 keeps its official split and any
   split-spanning group is moved into train (never out of it), so val/test stay strictly
   unseen and as close to the published split as possible.
5. Remove from the external set any image that near-duplicates a TN5000 image.

Augmentation and class re-balancing happen later and only on the train split.
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from utils.processing import prepare_ultrasound

KAGGLE_SLUG = "diveshzz/thyroid-cancer-classification-ultrasound-dataset"
TN5000_DOI = "10.6084/m9.figshare.28455641"
DATA_DIR = Path("data")

# Visual inspection: <=30 bits are the same frame, 30-45 (dHash) / <=50 (pHash) are adjacent frames of one sweep.
DUP_DIST = 30
GROUP_DIST = 45
PHASH_DIST = 50
SEED = 42


def paths_for(dataset):
    d = DATA_DIR / dataset
    return {"dir": d, "splits": d / "splits.csv", "cache": d / "cache_224.npz", "audit": d / "audit.json"}


# Kept for modules that only need the recipe-sweep split.
SPLITS_CSV = paths_for("kaggle")["splits"]


def kaggle_root():
    root = os.environ.get("THYROID_DATA")
    if root:
        return Path(root)
    # Prefer the local kagglehub cache: resolving through the API on every call gets rate-limited.
    cached = sorted((Path.home() / ".cache/kagglehub/datasets" / KAGGLE_SLUG / "versions").glob("*/Thyroid Data"))
    if cached:
        return cached[-1]
    import kagglehub
    return Path(kagglehub.dataset_download(KAGGLE_SLUG)) / "Thyroid Data"


# Backwards-compatible name.
dataset_root = kaggle_root


def tn5000_root():
    return Path(os.environ.get("TN5000_DATA", Path.home() / ".cache/tn5000/raw/TN5000_forReview"))


# ---------------------------------------------------------------- hashing / grouping
def dhash256(path):
    g = np.asarray(Image.open(path).convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def phash255(path):
    g = np.asarray(Image.open(path).convert("L").resize((64, 64), Image.BILINEAR), dtype=np.float32)
    coeffs = cv2.dct(g)[:16, :16].flatten()[1:]  # drop the DC term
    return coeffs > np.median(coeffs)


def hamming(a, b):
    """Pairwise Hamming distances between boolean code matrices via one matmul."""
    a, b = a.astype(np.float32), b.astype(np.float32)
    return (a.sum(1)[:, None] + b.sum(1)[None, :] - 2 * a @ b.T).round().astype(np.int32)


def _union_find_groups(adjacency):
    parent = list(range(len(adjacency)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in np.argwhere(np.triu(adjacency, k=1)):
        parent[find(i)] = find(j)
    return np.array([find(i) for i in range(len(adjacency))])


def hash_and_group(df):
    """Adds group / label_conflict columns; returns the codes for cross-dataset checks."""
    d = np.stack([dhash256(p) for p in df.path])
    p = np.stack([phash255(p) for p in df.path])
    dd, dp = hamming(d, d), hamming(p, p)
    np.fill_diagonal(dd, 999)
    np.fill_diagonal(dp, 999)
    df["group"] = _union_find_groups((dd <= GROUP_DIST) | (dp <= PHASH_DIST))
    df["dhash_group"] = _union_find_groups(dd <= GROUP_DIST)
    labels = df.label.values
    dup_pairs = np.argwhere(np.triu(dd <= DUP_DIST, k=1))
    conflict = {int(k) for i, j in dup_pairs if labels[i] != labels[j] for k in (i, j)}
    df["label_conflict"] = df.index.isin(conflict)
    stats = {"near_duplicate_pairs": int(len(dup_pairs)),
             "cross_class_duplicate_pairs": int(sum(labels[i] != labels[j] for i, j in dup_pairs))}
    return (d, p), stats


def _move_spanning_groups_to_train(df):
    kept = df.split.isin(["train", "val", "test"])
    spans = df[kept].groupby("group").split.nunique()
    moved = df.index[kept & df.group.isin(spans[spans > 1].index) & (df.split != "train")]
    df.loc[moved, "split"] = "train"
    assert (df[df.split.isin(["train", "val", "test"])].groupby("group").split.nunique() == 1).all()
    return [df.file[i] for i in moved]


def _split_counts(df, splits):
    return {s: {"n": int((df.split == s).sum()),
                "benign": int(((df.split == s) & (df.label == 0)).sum()),
                "malignant": int(((df.split == s) & (df.label == 1)).sum())} for s in splits}


# ---------------------------------------------------------------- indexing
def index_kaggle(root):
    rows = []
    for label in ("0", "1"):
        for name in sorted(os.listdir(root / label)):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                rows.append({"path": str(root / label / name), "label": int(label), "file": f"{label}/{name}"})
    return pd.DataFrame(rows)


def index_tn5000(root):
    official = {}
    for split in ("train", "val", "test"):
        for image_id in (root / "ImageSets/Main" / f"{split}.txt").read_text().split():
            official[image_id] = split
    rows = []
    for image_id, split in sorted(official.items()):
        obj = ET.parse(root / "Annotations" / f"{image_id}.xml").getroot().find("object")
        box = [int(obj.find(f"bndbox/{k}").text) for k in ("xmin", "ymin", "xmax", "ymax")]
        rows.append({"path": str(root / "JPEGImages" / f"{image_id}.jpg"), "label": int(obj.find("name").text),
                     "file": f"{image_id}.jpg", "official_split": split,
                     "xmin": box[0], "ymin": box[1], "xmax": box[2], "ymax": box[3]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- builders
def build_kaggle_dev():
    """
    Group-stratified 60/16/20 split of the Kaggle set (used for the recipe sweep).
    Folds are drawn on dHash groups and pHash links are merged afterwards (moving into
    train), which reproduces the exact split the recipe sweep was run on.
    """
    df = index_kaggle(kaggle_root())
    _, stats = hash_and_group(df)
    df["split"] = "excluded"
    clean = df[~df.label_conflict]
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    dev_idx, test_idx = next(outer.split(clean, clean.label, clean.dhash_group))
    df.loc[clean.index[test_idx], "split"] = "test"
    dev = clean.iloc[dev_idx]
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(inner.split(dev, dev.label, dev.dhash_group))
    df.loc[dev.index[tr_idx], "split"] = "train"
    df.loc[dev.index[va_idx], "split"] = "val"
    moved = _move_spanning_groups_to_train(df)
    audit = {"dataset": "kaggle", "images_total": int(len(df)),
             "class_counts": {str(k): int(v) for k, v in df.label.value_counts().sort_index().items()},
             **stats, "images_excluded_label_conflict": int(df.label_conflict.sum()),
             "scan_groups": int(df.group.nunique()), "multi_image_groups": int((df.group.value_counts() > 1).sum()),
             "moved_to_train_by_group_check": moved,
             "group_dist_bits": GROUP_DIST, "phash_dist_bits": PHASH_DIST, "dup_dist_bits": DUP_DIST,
             "splits": _split_counts(df, ("train", "val", "test"))}
    return df, audit


def build_tn5000():
    """Official TN5000 split made group-safe, plus the Kaggle set as an external test."""
    df = index_tn5000(tn5000_root())
    (td, tp), stats = hash_and_group(df)
    # How leaky is the published image-level split? Near-exact copies of an official-train image.
    near_exact = hamming(td, td) <= DUP_DIST
    np.fill_diagonal(near_exact, False)
    in_train = (df.official_split == "train").values
    leaked = {s: int((near_exact[(df.official_split == s).values][:, in_train].any(axis=1)).sum())
              for s in ("val", "test")}
    df["split"] = np.where(df.label_conflict, "excluded", df.official_split)
    moved = _move_spanning_groups_to_train(df)

    ext = index_kaggle(kaggle_root())
    (kd, kp), ext_stats = hash_and_group(ext)
    overlap = ((hamming(kd, td) <= GROUP_DIST) | (hamming(kp, tp) <= PHASH_DIST)).any(axis=1)
    ext["split"] = np.where(ext.label_conflict | overlap, "excluded", "external")
    ext["group"] = ext.group + df.group.max() + 1
    ext["file"] = "kaggle/" + ext.file

    audit = {
        "dataset": "tn5000", "source": f"figshare DOI {TN5000_DOI} (CC BY); labels biopsy-confirmed per authors",
        "images_total": int(len(df)),
        "class_counts": {str(k): int(v) for k, v in df.label.value_counts().sort_index().items()},
        **stats, "images_excluded_label_conflict": int(df.label_conflict.sum()),
        "scan_groups": int(df.group.nunique()), "multi_image_groups": int((df.group.value_counts() > 1).sum()),
        "moved_to_train_by_group_check": moved,
        "official_split_changes": {s: int(((df.official_split == s) & (df.split != s)).sum())
                                   for s in ("train", "val", "test")},
        "official_split_near_exact_copies_of_train": leaked,
        "group_dist_bits": GROUP_DIST, "phash_dist_bits": PHASH_DIST, "dup_dist_bits": DUP_DIST,
        "splits": _split_counts(df, ("train", "val", "test")),
        "external": {"source": f"Kaggle {KAGGLE_SLUG}", "images_total": int(len(ext)),
                     "excluded_label_conflict": int(ext.label_conflict.sum()),
                     "excluded_overlap_with_tn5000": int(overlap.sum()),
                     **ext_stats, **_split_counts(ext, ("external",))["external"]},
    }
    return pd.concat([df, ext], ignore_index=True), audit


def build_cache(df, out):
    """Runs the deterministic inference preprocessing once so train/serve are identical."""
    usable = df[df.split != "excluded"].reset_index(drop=True)
    images, boxes = [], []
    for row in usable.itertuples():
        if "xmin" in usable.columns and not np.isnan(row.xmin):
            img, to_frame = prepare_ultrasound(Image.open(row.path), return_transform=True)
            boxes.append([*to_frame(row.xmin, row.ymin), *to_frame(row.xmax, row.ymax)])
        else:
            img = prepare_ultrasound(Image.open(row.path))
            boxes.append([np.nan] * 4)
        images.append(img)
    np.savez_compressed(
        out,
        images=np.stack(images),
        labels=usable.label.values.astype(np.int8),
        split=usable.split.values.astype("U8"),
        files=usable.file.values.astype("U24"),
        boxes=np.array(boxes, dtype=np.float32),
    )


def load_cache(dataset="tn5000"):
    d = np.load(paths_for(dataset)["cache"])
    out = {}
    for s in ("train", "val", "test", "external"):
        m = d["split"] == s
        if m.any():
            out[s] = (d["images"][m], d["labels"][m].astype(np.float32), d["files"][m])
    out["test_boxes"] = d["boxes"][d["split"] == "test"]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["tn5000", "kaggle"], default="tn5000")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    p = paths_for(args.dataset)
    p["dir"].mkdir(parents=True, exist_ok=True)
    if p["splits"].exists() and p["cache"].exists() and not args.force:
        print(f"{p['splits']} and {p['cache']} already exist (use --force to rebuild)")
        return
    df, audit = build_tn5000() if args.dataset == "tn5000" else build_kaggle_dev()
    df.drop(columns=["path"]).to_csv(p["splits"], index=False)
    p["audit"].write_text(json.dumps(audit, indent=2))
    build_cache(df, p["cache"])
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
