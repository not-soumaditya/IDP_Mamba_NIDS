"""
UNSW-NB15 data cleaning, preprocessing, and PyTorch dataset for sequence models.

Two main components:
  1. clean_and_preprocess() — cleans raw CSVs, encodes features, scales, saves
  2. NIDSSequenceDataset — sliding-window PyTorch Dataset for sequential models

Usage (standalone):
    python model/dataset.py          # runs cleaning pipeline
"""

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"


# ──────────────────────────────────────────────────────────
# 1. CLEANING & PREPROCESSING
# ──────────────────────────────────────────────────────────

CATEGORICAL_COLS = ["proto", "service", "state"]


def clean_and_preprocess(raw_dir: Path = RAW_DIR,
                         processed_dir: Path = PROCESSED_DIR,
                         docs_dir: Path = DOCS_DIR):
    """
    Full cleaning pipeline:
      - Load raw train/test CSVs
      - Drop 'id' column
      - Standardize attack_cat
      - Handle missing values
      - LabelEncode categoricals
      - StandardScaler on numerics
      - Encode attack_cat -> attack_label (0-9)
      - Save processed data + artifacts
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ───────────────────────────────────────────────
    train_path = raw_dir / "UNSW_NB15_training-set.csv"
    test_path = raw_dir / "UNSW_NB15_testing-set.csv"

    if not train_path.exists() or not test_path.exists():
        print(f"ERROR: Raw CSV files not found in {raw_dir}")
        print("Run  python data/download_data.py  first.")
        sys.exit(1)

    print("Loading raw CSVs...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"  Train: {train_df.shape}   Test: {test_df.shape}")

    # ── Drop id column ─────────────────────────────────────
    for df in [train_df, test_df]:
        if "id" in df.columns:
            df.drop(columns=["id"], inplace=True)

    # ── Standardize attack_cat ─────────────────────────────
    for df in [train_df, test_df]:
        df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
        # Fill blanks / NaN / 'nan' with 'Normal' where label == 0
        mask_normal = (df["label"] == 0) | (df["attack_cat"].isin(["", "nan", "NaN", " "]))
        df.loc[mask_normal, "attack_cat"] = "Normal"
        # Also fix any remaining blanks
        df["attack_cat"] = df["attack_cat"].replace({"": "Normal", "nan": "Normal",
                                                       " ": "Normal", "NaN": "Normal"})

    # ── Handle missing values ──────────────────────────────
    for df in [train_df, test_df]:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in num_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
                
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in cat_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)

    print("  Missing values handled.")

    # ── LabelEncode categoricals ───────────────────────────
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str)])
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        label_encoders[col] = le
        print(f"  Encoded '{col}': {len(le.classes_)} categories")

    # ── Encode attack_cat -> attack_label ──────────────────
    attack_le = LabelEncoder()
    combined_attack = pd.concat([train_df["attack_cat"], test_df["attack_cat"]])
    attack_le.fit(combined_attack)
    train_df["attack_label"] = attack_le.transform(train_df["attack_cat"])
    test_df["attack_label"] = attack_le.transform(test_df["attack_cat"])
    label_encoders["attack_cat"] = attack_le

    num_classes = len(attack_le.classes_)
    print(f"  Attack categories ({num_classes}): {list(attack_le.classes_)}")

    # ── StandardScaler on numeric features ─────────────────
    # Exclude target columns from scaling
    exclude_cols = {"label", "attack_cat", "attack_label"}
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    numeric_feature_cols = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    train_df[numeric_feature_cols] = scaler.fit_transform(train_df[numeric_feature_cols])
    test_df[numeric_feature_cols] = scaler.transform(test_df[numeric_feature_cols])
    print(f"  Scaled {len(numeric_feature_cols)} numeric features.")

    # ── Save processed data ────────────────────────────────
    train_df.to_csv(processed_dir / "train_processed.csv", index=False)
    test_df.to_csv(processed_dir / "test_processed.csv", index=False)
    print(f"  Saved processed data to {processed_dir}")

    # Save encoders and scaler for later use
    artifacts = {
        "label_encoders": label_encoders,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "numeric_feature_cols": numeric_feature_cols,
        "num_classes": num_classes,
        "attack_classes": list(attack_le.classes_),
    }
    with open(processed_dir / "preprocessing_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    # ── Class distribution summary ─────────────────────────
    _save_class_distribution(train_df, test_df, attack_le, docs_dir)

    return train_df, test_df, artifacts


def _save_class_distribution(train_df, test_df, attack_le, docs_dir):
    """Generate class distribution table and chart for presentation slides."""
    # Build summary table
    train_counts = train_df["attack_cat"].value_counts()
    test_counts = test_df["attack_cat"].value_counts()

    summary = pd.DataFrame({
        "Category": attack_le.classes_,
    })
    summary["Train_Count"] = summary["Category"].map(
        lambda c: train_counts.get(c, 0))
    summary["Test_Count"] = summary["Category"].map(
        lambda c: test_counts.get(c, 0))
    summary["Train_%"] = (summary["Train_Count"] / summary["Train_Count"].sum() * 100).round(2)
    summary["Test_%"] = (summary["Test_Count"] / summary["Test_Count"].sum() * 100).round(2)
    summary = summary.sort_values("Train_Count", ascending=False).reset_index(drop=True)

    # Save markdown table
    md_lines = ["# UNSW-NB15 Class Distribution\n"]
    md_lines.append("| Category | Train Count | Train % | Test Count | Test % |")
    md_lines.append("|----------|-------------|---------|------------|--------|")
    for _, row in summary.iterrows():
        md_lines.append(
            f"| {row['Category']} | {int(row['Train_Count']):,} | {row['Train_%']:.2f}% "
            f"| {int(row['Test_Count']):,} | {row['Test_%']:.2f}% |"
        )
    md_lines.append(f"\n**Total:** Train={int(summary['Train_Count'].sum()):,}, "
                     f"Test={int(summary['Test_Count'].sum()):,}")

    with open(docs_dir / "class_distribution.md", "w") as f:
        f.write("\n".join(md_lines))

    # Save chart (PNG for slides)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = sns.color_palette("husl", len(summary))

    for ax, split, col in zip(axes, ["Train", "Test"], ["Train_Count", "Test_Count"]):
        bars = ax.barh(summary["Category"], summary[col], color=colors)
        ax.set_xlabel("Count")
        ax.set_title(f"{split} Set Distribution")
        ax.invert_yaxis()
        for bar, val in zip(bars, summary[col]):
            ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                    f"{int(val):,}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(docs_dir / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Class distribution saved to {docs_dir}/class_distribution.md and .png")

    # Print to console
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))
    print()

    return summary


def load_processed_data(processed_dir: Path = PROCESSED_DIR):
    """Load already-processed train/test DataFrames and artifacts."""
    train_df = pd.read_csv(processed_dir / "train_processed.csv")
    test_df = pd.read_csv(processed_dir / "test_processed.csv")
    with open(processed_dir / "preprocessing_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    return train_df, test_df, artifacts


# ──────────────────────────────────────────────────────────
# 2. SEQUENCE DATASET
# ──────────────────────────────────────────────────────────

class NIDSSequenceDataset(Dataset):
    """
    Sliding-window sequence dataset for network flow data.

    Why this matters: Mamba needs to see a *sequence* of traffic to detect patterns, 
    not just a single snapshot. We use a sliding window to group consecutive flows.
    Each sample is a window of `seq_len` consecutive flows.
    The label is the attack_label of the LAST flow in the window.
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list,
                 seq_len: int = 32, stride: int = 1):
        """
        Args:
            df: Processed DataFrame with features and 'attack_label' column.
            feature_cols: List of feature column names to use.
            seq_len: Number of consecutive flows per sequence (our context window).
            stride: Step size for sliding window (1 means we shift by 1 packet each time).
        """
        self.seq_len = seq_len
        self.stride = stride

        # Extract feature matrix and labels for faster tensor conversion
        features = df[feature_cols].values.astype(np.float32)
        labels = df["attack_label"].values.astype(np.int64)

        # Build index pairs for sliding window
        self.indices = []
        for i in range(0, len(features) - seq_len + 1, stride):
            self.indices.append(i)

        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Fetch the specific window of packets
        start = self.indices[idx]
        end = start + self.seq_len

        # Convert the window to a PyTorch tensor
        x = torch.tensor(self.features[start:end], dtype=torch.float32)
        
        # The label for the entire sequence is the label of the final packet in the window
        y = torch.tensor(self.labels[end - 1], dtype=torch.long)
        return x, y


def build_dataloaders(train_df, test_df, feature_cols,
                      seq_len=32, stride=1, batch_size=64,
                      max_train_sequences=None):
    """
    Build train and test DataLoaders from processed DataFrames.

    Args:
        max_train_sequences: If set, limit training data for quick prototyping.
    """
    train_dataset = NIDSSequenceDataset(train_df, feature_cols,
                                         seq_len=seq_len, stride=stride)
    test_dataset = NIDSSequenceDataset(test_df, feature_cols,
                                        seq_len=seq_len, stride=stride)

    if max_train_sequences and len(train_dataset) > max_train_sequences:
        # Use a random subset for prototyping
        indices = np.random.RandomState(42).choice(
            len(train_dataset), max_train_sequences, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)

    print(f"  Train sequences: {len(train_dataset):,}")
    print(f"  Test sequences:  {len(test_dataset):,}")
    print(f"  Batch size: {batch_size}")

    return train_loader, test_loader


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("UNSW-NB15 DATA CLEANING & PREPROCESSING")
    print("=" * 60)
    train_df, test_df, artifacts = clean_and_preprocess()

    print("\n" + "=" * 60)
    print("BUILDING SEQUENCE DATASETS")
    print("=" * 60)
    feature_cols = artifacts["feature_cols"]
    train_loader, test_loader = build_dataloaders(
        train_df, test_df, feature_cols, seq_len=32, stride=1, batch_size=64)

    # Quick sanity check
    x_batch, y_batch = next(iter(train_loader))
    print(f"\n  Sample batch shapes: X={x_batch.shape}, Y={y_batch.shape}")
    print(f"  Feature dim: {x_batch.shape[-1]}")
    print(f"  Num classes: {artifacts['num_classes']}")
    print("\nDone!")
