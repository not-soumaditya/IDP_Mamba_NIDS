"""
Train the CPU Mamba prototype on a subset of UNSW-NB15 sequence data.

Goal: PROVE the forward/backward pass works and loss decreases.
NOT intended to produce a great model — that's what the GPU notebook is for.

Usage:
    python model/train_prototype.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
CHECKPOINTS_DIR = PROJECT_ROOT / "model" / "checkpoints"

sys.path.insert(0, str(PROJECT_ROOT))
from model.dataset import load_processed_data, build_dataloaders
from model.mamba_minimal import MambaNIDS
from model.evaluate import evaluate_model, benchmark_mamba_inference, write_prototype_results


def train_prototype(
    n_epochs: int = 5,
    max_train_sequences: int = 8000,
    seq_len: int = 32,
    batch_size: int = 64,
    d_model: int = 64,
    n_layers: int = 2,
    d_state: int = 16,
    lr: float = 1e-3,
):
    """Train the CPU Mamba prototype."""
    print("=" * 60)
    print("CPU MAMBA PROTOTYPE TRAINING")
    print("=" * 60)

    # Load data
    print("\nLoading processed data...")
    train_df, test_df, artifacts = load_processed_data()
    feature_cols = artifacts["feature_cols"]
    num_classes = artifacts["num_classes"]
    class_names = artifacts["attack_classes"]
    input_dim = len(feature_cols)

    print(f"  Input dim: {input_dim}, Num classes: {num_classes}")

    # Build dataloaders (limited subset for CPU training)
    print(f"\nBuilding sequence datasets (max {max_train_sequences:,} train sequences)...")
    train_loader, test_loader = build_dataloaders(
        train_df, test_df, feature_cols,
        seq_len=seq_len, stride=1, batch_size=batch_size,
        max_train_sequences=max_train_sequences,
    )

    # Build model
    print(f"\nBuilding MambaNIDS (d_model={d_model}, n_layers={n_layers}, d_state={d_state})...")
    model = MambaNIDS(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
    )
    print(f"  Parameters: {model.count_parameters():,}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # Training loop
    print(f"\nTraining for {n_epochs} epochs on CPU...")
    epoch_losses = []
    epoch_accs = []

    total_train_start = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        epoch_start = time.time()

        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
            batch_count += 1

            if (batch_idx + 1) % 20 == 0:
                print(f"    Batch {batch_idx+1}/{len(train_loader)}: "
                      f"loss={loss.item():.4f}")

        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / batch_count
        acc = correct / total
        epoch_losses.append(avg_loss)
        epoch_accs.append(acc)

        scheduler.step()

        print(f"  Epoch {epoch}/{n_epochs}: loss={avg_loss:.4f}, "
              f"acc={acc:.4f}, time={epoch_time:.1f}s")

    total_train_time = time.time() - total_train_start
    print(f"\nTotal training time: {total_train_time:.1f}s")

    # Save training loss plot
    _save_training_plot(epoch_losses, epoch_accs, DOCS_DIR)

    # Save checkpoint
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINTS_DIR / "mamba_prototype_cpu.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": n_epochs,
        "loss": epoch_losses[-1],
        "config": {
            "input_dim": input_dim,
            "num_classes": num_classes,
            "d_model": d_model,
            "n_layers": n_layers,
            "d_state": d_state,
        },
    }, checkpoint_path)
    print(f"  Checkpoint saved to {checkpoint_path}")

    # Evaluate on test set (use a subset for speed)
    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, test_loader, class_names, device="cpu",
                             max_batches=50)

    # Benchmark inference time
    inference_time = benchmark_mamba_inference(model, input_dim, seq_len, n_runs=50)

    # Write prototype results
    write_prototype_results(
        metrics=metrics,
        class_names=class_names,
        epoch_losses=epoch_losses,
        epoch_accs=epoch_accs,
        train_time=total_train_time,
        inference_time=inference_time,
        model_params=model.count_parameters(),
        config={
            "d_model": d_model,
            "n_layers": n_layers,
            "d_state": d_state,
            "seq_len": seq_len,
            "max_train_sequences": max_train_sequences,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "lr": lr,
        },
        docs_dir=DOCS_DIR,
    )

    # Check that loss decreased
    if len(epoch_losses) >= 2 and epoch_losses[-1] < epoch_losses[0]:
        print("\n✓ Loss decreased from {:.4f} to {:.4f} — forward/backward pass works!".format(
            epoch_losses[0], epoch_losses[-1]))
    else:
        print("\n⚠ Loss did not clearly decrease — check model/data")

    print("\n" + "=" * 60)
    print("PROTOTYPE TRAINING COMPLETE")
    print("=" * 60)


def _save_training_plot(losses, accs, docs_dir):
    """Save training loss and accuracy plots."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(losses) + 1)

    ax1.plot(epochs, losses, "b-o", linewidth=2, markersize=6)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("CPU Mamba Prototype — Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, accs, "g-o", linewidth=2, markersize=6)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training Accuracy")
    ax2.set_title("CPU Mamba Prototype — Training Accuracy")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(docs_dir / "prototype_training_curves.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved to {docs_dir / 'prototype_training_curves.png'}")


if __name__ == "__main__":
    train_prototype()
