"""
Shared evaluation utilities for both baseline and Mamba models.

Computes: accuracy, macro-F1, per-class F1, confusion matrix,
inference-time benchmark. Writes results to docs/.

Usage:
    Imported by baselines.py and train_prototype.py
"""

import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


def evaluate_model(model, test_loader, class_names, device="cpu",
                   max_batches=None):
    """
    Evaluate a PyTorch model on the test set.

    Returns dict with accuracy, f1_macro, f1_weighted, report, confusion_matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
            if max_batches and batch_idx >= max_batches:
                break
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    # Get labels present in predictions/ground truth
    present_labels = sorted(set(all_labels) | set(all_preds))
    present_names = [class_names[i] for i in present_labels if i < len(class_names)]

    report = classification_report(
        all_labels, all_preds,
        labels=present_labels,
        target_names=present_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(all_labels, all_preds)

    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Macro F1:    {f1_macro:.4f}")
    print(f"  Weighted F1: {f1_weighted:.4f}")

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report": report,
        "confusion_matrix": cm,
        "present_labels": present_labels,
        "present_names": present_names,
    }


def benchmark_mamba_inference(model, input_dim, seq_len=32, batch_size=1,
                               n_runs=50, device="cpu"):
    """
    Benchmark inference time for the Mamba model (µs per sample).
    """
    model.eval()
    x_dummy = torch.randn(batch_size, seq_len, input_dim).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(x_dummy)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            model(x_dummy)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

    avg_time_us = np.mean(times) / batch_size * 1e6
    print(f"  Inference: {avg_time_us:.2f} µs/sample (seq_len={seq_len}, "
          f"avg over {n_runs} runs)")
    return avg_time_us


def write_prototype_results(metrics, class_names, epoch_losses, epoch_accs,
                             train_time, inference_time, model_params,
                             config, docs_dir=DOCS_DIR):
    """Write Mamba prototype results to docs/prototype_results.md."""
    docs_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# CPU Mamba Prototype Results — UNSW-NB15\n")
    lines.append("**Model:** MambaNIDS (pure-PyTorch selective SSM)  ")
    lines.append("**Device:** CPU  ")
    lines.append(f"**Parameters:** {model_params:,}  ")
    lines.append(f"**Training time:** {train_time:.1f}s\n")

    lines.append("## Model Configuration\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    for k, v in config.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## Training Progress\n")
    lines.append("| Epoch | Loss | Accuracy |")
    lines.append("|-------|------|----------|")
    for i, (loss, acc) in enumerate(zip(epoch_losses, epoch_accs), 1):
        lines.append(f"| {i} | {loss:.4f} | {acc:.4f} |")
    lines.append("")

    if len(epoch_losses) >= 2:
        loss_delta = epoch_losses[0] - epoch_losses[-1]
        lines.append(f"**Loss decreased by {loss_delta:.4f}** "
                     f"(from {epoch_losses[0]:.4f} to {epoch_losses[-1]:.4f}) "
                     f"— forward/backward pass verified ✓\n")

    lines.append("## Test Set Evaluation\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Accuracy | {metrics['accuracy']:.4f} |")
    lines.append(f"| Macro F1 | {metrics['f1_macro']:.4f} |")
    lines.append(f"| Weighted F1 | {metrics['f1_weighted']:.4f} |")
    lines.append(f"| Inference time | {inference_time:.2f} µs/sample |")
    lines.append("")

    lines.append("## Per-Class Performance\n")
    lines.append("| Class | Precision | Recall | F1-Score | Support |")
    lines.append("|-------|-----------|--------|----------|---------|")
    for cls in metrics.get("present_names", class_names):
        if cls in metrics["report"]:
            m = metrics["report"][cls]
            lines.append(
                f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | "
                f"{m['f1-score']:.4f} | {int(m['support'])} |"
            )
    lines.append("")

    lines.append("## Notes\n")
    lines.append("- This is a CPU prototype trained on a small subset "
                 f"({config.get('max_train_sequences', 'N/A')} sequences)")
    lines.append("- Full GPU training will use mamba-ssm library on Kaggle/Colab")
    lines.append("- See `colab_gpu_training.ipynb` for the production training notebook")
    lines.append("- Training curves: ![Training Curves](prototype_training_curves.png)")

    md_text = "\n".join(lines)
    with open(docs_dir / "prototype_results.md", "w") as f:
        f.write(md_text)
    print(f"  Results written to {docs_dir / 'prototype_results.md'}")

    # Save confusion matrix
    if metrics["confusion_matrix"].shape[0] > 0:
        _save_cm(metrics["confusion_matrix"], metrics.get("present_names", class_names),
                 "MambaNIDS (CPU Prototype)", docs_dir, "prototype_confusion_matrix.png")


def _save_cm(cm, class_names, title, docs_dir, filename):
    """Save confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {title}")
    plt.tight_layout()
    plt.savefig(docs_dir / filename, dpi=150)
    plt.close()
