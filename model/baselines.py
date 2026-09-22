"""
Classical ML baselines for UNSW-NB15 (flat features, no sequences).

Trains RandomForest (and optionally XGBoost) on the cleaned tabular data,
evaluates accuracy, macro-F1, per-class F1, confusion matrix, and
benchmarks inference time. Writes results to docs/baseline_results.md.

Usage:
    python model/baselines.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix,
)
import pickle
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"
CHECKPOINTS_DIR = PROJECT_ROOT / "model" / "checkpoints"

sys.path.insert(0, str(PROJECT_ROOT))
from model.dataset import load_processed_data


def train_random_forest(X_train, y_train, X_test, y_test, class_names):
    """Train and evaluate a RandomForestClassifier as our baseline."""
    print("Training RandomForest classifier...")
    
    # Initialize Random Forest with limited depth to prevent overfitting
    # n_jobs=-1 uses all available CPU cores for faster training
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )

    # Track training time
    start_time = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"  Training time: {train_time:.2f}s")

    # Generate predictions for the test set
    y_pred = rf.predict(X_test)

    # Calculate core metrics. Macro F1 is crucial because the dataset is highly imbalanced.
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Macro F1:    {f1_macro:.4f}")
    print(f"  Weighted F1: {f1_weighted:.4f}")

    # Generate detailed per-class breakdown
    report = classification_report(y_test, y_pred, target_names=class_names,
                                    output_dict=True, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)

    # Run speed benchmark to compare against Mamba later
    inference_time = benchmark_inference(rf, X_test)

    # Save the trained model for future use
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINTS_DIR / "random_forest.pkl", "wb") as f:
        pickle.dump(rf, f)
    print(f"  Model saved to {CHECKPOINTS_DIR / 'random_forest.pkl'}")

    return {
        "model_name": "RandomForest",
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "train_time": train_time,
        "inference_time_per_sample_us": inference_time,
        "report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred,
    }


def benchmark_inference(model, X_test, n_runs=100):
    """Benchmark average inference time per sample (microseconds) to evaluate real-time capability."""
    # Use a small batch for benchmarking to simulate real-time packet arrival
    X_sample = X_test[:100] if len(X_test) > 100 else X_test

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(X_sample)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time_per_sample = np.mean(times) / len(X_sample) * 1e6  # microseconds
    print(f"  Inference: {avg_time_per_sample:.2f} µs/sample (avg over {n_runs} runs)")
    return avg_time_per_sample


def write_results_markdown(results, class_names, docs_dir=DOCS_DIR):
    """Write baseline results to docs/baseline_results.md."""
    docs_dir.mkdir(parents=True, exist_ok=True)

    r = results
    lines = []
    lines.append("# Baseline Model Results — UNSW-NB15\n")
    lines.append(f"**Model:** {r['model_name']}  ")
    lines.append(f"**Dataset:** UNSW-NB15 (flat features, non-sequential)  ")
    lines.append(f"**Training time:** {r['train_time']:.2f}s\n")

    lines.append("## Overall Metrics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Accuracy | {r['accuracy']:.4f} |")
    lines.append(f"| Macro F1 | {r['f1_macro']:.4f} |")
    lines.append(f"| Weighted F1 | {r['f1_weighted']:.4f} |")
    lines.append(f"| Inference time | {r['inference_time_per_sample_us']:.2f} µs/sample |")
    lines.append("")

    lines.append("## Per-Class Performance\n")
    lines.append("| Class | Precision | Recall | F1-Score | Support |")
    lines.append("|-------|-----------|--------|----------|---------|")
    for cls in class_names:
        if cls in r["report"]:
            m = r["report"][cls]
            lines.append(
                f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | "
                f"{m['f1-score']:.4f} | {int(m['support'])} |"
            )
    lines.append("")

    md_text = "\n".join(lines)
    with open(docs_dir / "baseline_results.md", "w") as f:
        f.write(md_text)

    # Save confusion matrix plot
    _save_confusion_matrix(r["confusion_matrix"], class_names,
                           r["model_name"], docs_dir)

    print(f"  Results written to {docs_dir / 'baseline_results.md'}")


def _save_confusion_matrix(cm, class_names, model_name, docs_dir):
    """Save confusion matrix as PNG."""
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(docs_dir / "baseline_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {docs_dir / 'baseline_confusion_matrix.png'}")


def main():
    print("=" * 60)
    print("BASELINE MODEL TRAINING — UNSW-NB15")
    print("=" * 60)

    # Load processed data
    train_df, test_df, artifacts = load_processed_data()
    feature_cols = artifacts["feature_cols"]
    class_names = artifacts["attack_classes"]
    num_classes = artifacts["num_classes"]

    # Prepare flat features (no sequences)
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["attack_label"].values
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["attack_label"].values

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Classes: {num_classes} — {class_names}")

    # Train RandomForest
    print()
    rf_results = train_random_forest(X_train, y_train, X_test, y_test, class_names)

    # Write results
    print()
    write_results_markdown(rf_results, class_names)

    print("\n" + "=" * 60)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
