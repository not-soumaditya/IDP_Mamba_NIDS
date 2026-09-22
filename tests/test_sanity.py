"""
Sanity tests for NIDS-Mamba project.

Tests:
  1. NIDSSequenceDataset produces correctly shaped tensors
  2. MambaNIDS forward pass runs without error and outputs correct shape
  3. A training step actually reduces loss over a few iterations

Run:
    pytest tests/test_sanity.py -v
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.mamba_minimal import MambaNIDS, SelectiveSSMBlock
from model.dataset import NIDSSequenceDataset


# ──────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────

@pytest.fixture
def dummy_df():
    """Create a small dummy DataFrame for testing."""
    import pandas as pd
    n_samples = 200
    n_features = 10
    np.random.seed(42)

    data = np.random.randn(n_samples, n_features).astype(np.float32)
    cols = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=cols)
    df["attack_label"] = np.random.randint(0, 10, n_samples)
    return df, cols


@pytest.fixture
def model_config():
    """Default model configuration for testing."""
    return {
        "input_dim": 10,
        "num_classes": 10,
        "d_model": 32,
        "n_layers": 2,
        "d_state": 8,
        "d_conv": 4,
        "expand": 2,
    }


# ──────────────────────────────────────────────────────────
# Test 1: Dataset shapes
# ──────────────────────────────────────────────────────────

class TestNIDSSequenceDataset:
    def test_correct_shape(self, dummy_df):
        """Test that dataset produces tensors with correct shapes."""
        df, feature_cols = dummy_df
        seq_len = 16
        dataset = NIDSSequenceDataset(df, feature_cols, seq_len=seq_len, stride=1)

        assert len(dataset) > 0, "Dataset should not be empty"

        x, y = dataset[0]
        assert x.shape == (seq_len, len(feature_cols)), \
            f"Expected ({seq_len}, {len(feature_cols)}), got {x.shape}"
        assert y.shape == (), f"Expected scalar label, got shape {y.shape}"
        assert x.dtype == torch.float32, f"Expected float32, got {x.dtype}"
        assert y.dtype == torch.long, f"Expected long, got {y.dtype}"

    def test_correct_length(self, dummy_df):
        """Test that dataset length matches expected number of windows."""
        df, feature_cols = dummy_df
        seq_len = 16
        stride = 1
        dataset = NIDSSequenceDataset(df, feature_cols, seq_len=seq_len, stride=stride)

        expected_len = (len(df) - seq_len) // stride + 1
        assert len(dataset) == expected_len, \
            f"Expected {expected_len} windows, got {len(dataset)}"

    def test_stride(self, dummy_df):
        """Test that stride parameter works correctly."""
        df, feature_cols = dummy_df
        seq_len = 16
        stride = 4
        dataset = NIDSSequenceDataset(df, feature_cols, seq_len=seq_len, stride=stride)

        expected_len = (len(df) - seq_len) // stride + 1
        assert len(dataset) == expected_len

    def test_label_is_last_flow(self, dummy_df):
        """Test that the label is from the last flow in the window."""
        df, feature_cols = dummy_df
        seq_len = 16
        dataset = NIDSSequenceDataset(df, feature_cols, seq_len=seq_len, stride=1)

        _, y = dataset[0]
        expected_label = df["attack_label"].iloc[seq_len - 1]
        assert y.item() == expected_label, \
            f"Expected label {expected_label}, got {y.item()}"


# ──────────────────────────────────────────────────────────
# Test 2: MambaNIDS forward pass
# ──────────────────────────────────────────────────────────

class TestMambaNIDS:
    def test_forward_pass_shape(self, model_config):
        """Test that forward pass produces correct output shape."""
        model = MambaNIDS(**model_config)
        batch_size = 4
        seq_len = 32

        x = torch.randn(batch_size, seq_len, model_config["input_dim"])
        logits = model(x)

        assert logits.shape == (batch_size, model_config["num_classes"]), \
            f"Expected ({batch_size}, {model_config['num_classes']}), got {logits.shape}"

    def test_forward_pass_no_error(self, model_config):
        """Test that forward pass runs without errors."""
        model = MambaNIDS(**model_config)
        x = torch.randn(2, 16, model_config["input_dim"])

        try:
            logits = model(x)
        except Exception as e:
            pytest.fail(f"Forward pass raised an exception: {e}")

    def test_output_is_finite(self, model_config):
        """Test that model output contains no NaN or Inf values."""
        model = MambaNIDS(**model_config)
        x = torch.randn(4, 32, model_config["input_dim"])
        logits = model(x)

        assert torch.isfinite(logits).all(), "Output contains NaN or Inf values"

    def test_different_seq_lengths(self, model_config):
        """Test that model handles various sequence lengths."""
        model = MambaNIDS(**model_config)

        for seq_len in [8, 16, 32, 64]:
            x = torch.randn(2, seq_len, model_config["input_dim"])
            logits = model(x)
            assert logits.shape == (2, model_config["num_classes"])

    def test_parameter_count(self, model_config):
        """Test that model has a reasonable number of parameters."""
        model = MambaNIDS(**model_config)
        n_params = model.count_parameters()
        assert n_params > 0, "Model should have parameters"
        assert n_params < 1_000_000, f"Too many parameters ({n_params:,}) for this config"


# ──────────────────────────────────────────────────────────
# Test 3: Training step reduces loss
# ──────────────────────────────────────────────────────────

class TestTrainingStep:
    def test_loss_decreases(self, model_config):
        """Test that loss decreases over a few training iterations."""
        torch.manual_seed(42)

        model = MambaNIDS(**model_config)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Create a tiny synthetic batch
        batch_size = 8
        seq_len = 16
        x = torch.randn(batch_size, seq_len, model_config["input_dim"])
        y = torch.randint(0, model_config["num_classes"], (batch_size,))

        # Train for a few steps on the same batch (should overfit)
        losses = []
        for _ in range(20):
            model.train()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0], \
            f"Loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"

    def test_gradients_flow(self, model_config):
        """Test that gradients flow through all parameters."""
        model = MambaNIDS(**model_config)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(4, 16, model_config["input_dim"])
        y = torch.randint(0, model_config["num_classes"], (4,))

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                # At least some gradients should be non-zero
                # (not checking all, as some might legitimately be zero)


class TestSelectiveSSMBlock:
    def test_forward_shape(self):
        """Test SSM block output shape."""
        d_model = 32
        block = SelectiveSSMBlock(d_model=d_model, d_state=8)
        x = torch.randn(2, 16, d_model)
        y = block(x)
        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"

    def test_causal(self):
        """Test that the SSM block is causal (output at t depends only on t and before)."""
        d_model = 16
        block = SelectiveSSMBlock(d_model=d_model, d_state=4, d_conv=4)
        block.eval()

        x1 = torch.randn(1, 10, d_model)
        x2 = x1.clone()
        # Modify x2 at position 8 (near the end)
        x2[0, 8, :] = torch.randn(d_model)

        with torch.no_grad():
            y1 = block(x1)
            y2 = block(x2)

        # Outputs before position 8 should be the same
        # (accounting for conv kernel size - outputs up to position 8-d_conv should match)
        # With d_conv=4, positions 0 through 4 should definitely be unaffected
        assert torch.allclose(y1[0, :4], y2[0, :4], atol=1e-5), \
            "SSM block is not causal — early outputs changed when later inputs changed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
