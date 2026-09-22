"""
Mamba-based Network Intrusion Detection System (NIDS) Model.

Architecture:
    Input → Linear Projection → [Mamba Block × N] → Global Avg Pool → Classifier

Supports two backends:
    1. Official mamba-ssm library (GPU-accelerated, requires NVIDIA CUDA)
    2. Pure PyTorch fallback (runs on any hardware)
"""
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────────
# Try to import the official mamba-ssm library
# ──────────────────────────────────────────────────────────────────
USING_MAMBA_SSM = False
try:
    from mamba_ssm import Mamba as MambaBlock
    USING_MAMBA_SSM = True
except ImportError:
    MambaBlock = None


# ──────────────────────────────────────────────────────────────────
# Pure PyTorch Selective SSM (fallback when mamba-ssm is unavailable)
# ──────────────────────────────────────────────────────────────────
class SelectiveSSM(nn.Module):
    """
    A selective State Space Model block implemented in pure PyTorch.
    Replicates the core ideas of Mamba without custom CUDA kernels.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = d_model * expand
        self.d_inner = d_inner
        self.d_state = d_state

        # Input projections (two branches for gating)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # 1D depthwise causal convolution
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
        )

        # Selective SSM parameters (input-dependent B, C, delta)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # A parameter in log-space for numerical stability
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # Two branches
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Causal convolution
        x_conv = x_branch.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_branch = torch.nn.functional.silu(x_conv.transpose(1, 2))

        # Compute input-dependent B, C, delta
        x_proj = self.x_proj(x_branch)
        B = x_proj[:, :, : self.d_state]
        C = x_proj[:, :, self.d_state : 2 * self.d_state]
        delta = torch.nn.functional.softplus(x_proj[:, :, -1:])
        delta = self.dt_proj(delta)

        # Discretize A
        A = -torch.exp(self.A_log)

        # Selective scan (sequential)
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device)
        outputs = []
        for t in range(seq_len):
            dt = delta[:, t, :]
            A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
            B_bar = dt.unsqueeze(-1) * B[:, t, :].unsqueeze(1)
            h = A_bar * h + B_bar * x_branch[:, t, :].unsqueeze(-1)
            y_t = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1)
            y_t = y_t + self.D * x_branch[:, t, :]
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)

        # Gating with z branch
        z = torch.nn.functional.silu(z)
        return self.out_proj(y * z)


# ──────────────────────────────────────────────────────────────────
# Main Model
# ──────────────────────────────────────────────────────────────────
class MambaNIDS(nn.Module):
    """
    Mamba-based Network Intrusion Detection Model.

    Args:
        input_dim:   Number of input features per flow (e.g., 42)
        d_model:     Model hidden dimension (default: 64)
        d_state:     SSM state expansion factor (default: 16)
        d_conv:      Local convolution width (default: 4)
        expand:      Block expansion factor (default: 2)
        n_layers:    Number of stacked Mamba blocks (default: 4)
        num_classes: Number of output classes (default: 10)
        dropout:     Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        input_dim,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        n_layers=4,
        num_classes=10,
        dropout=0.1,
    ):
        super().__init__()

        # Project input features to model dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Stack of Mamba blocks with LayerNorm
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            if USING_MAMBA_SSM and MambaBlock is not None:
                self.layers.append(
                    MambaBlock(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                    )
                )
            else:
                self.layers.append(
                    SelectiveSSM(d_model, d_state, d_conv, expand)
                )
            self.norms.append(nn.LayerNorm(d_model))

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self._backend = "mamba-ssm (GPU)" if USING_MAMBA_SSM else "PyTorch (CPU/GPU)"

    @property
    def backend(self):
        return self._backend

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) — sequence of network flows
        Returns:
            logits: (batch, num_classes)
        """
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = norm(x)
            x = layer(x)
            x = x + residual  # Residual connection

        x = x.mean(dim=1)     # Global average pooling
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits
