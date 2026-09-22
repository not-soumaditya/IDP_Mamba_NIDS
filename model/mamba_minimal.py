"""
Minimal pure-PyTorch Selective State Space Model (Mamba) block + NIDS classifier.

This is a simplified "mamba-minimal" style implementation that runs on CPU.
It implements the core selective SSM mechanism:
  input projection -> 1D causal conv -> SiLU -> selective SSM scan
  (with input-dependent B, C, delta) -> gated output projection

Architecture reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling
with Selective State Spaces" (2023).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSMBlock(nn.Module):
    """
    Minimal Selective State Space Model block (Mamba-style).

    Input: (batch, seq_len, d_model)
    Output: (batch, seq_len, d_model)

    Internal flow:
      1. Input projection: d_model -> 2*d_inner (split into x and z branches)
      2. x branch: 1D causal conv -> SiLU -> selective SSM scan
      3. z branch: SiLU gate
      4. Output = x_ssm * z_gate, projected back to d_model
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_rank: int = None):
        """
        Args:
            d_model: Input/output dimension.
            d_state: SSM state dimension (N in the paper).
            d_conv: Causal convolution kernel size.
            expand: Expansion factor for inner dimension.
            dt_rank: Rank for delta (dt) projection. Defaults to ceil(d_model/16).
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        # Input projection: project to 2*d_inner (x and z branches)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # 1D causal convolution on x branch
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,  # causal: we'll truncate
            groups=self.d_inner,  # depthwise
            bias=True,
        )

        # SSM parameters projection from x
        # Projects x -> (dt, B, C) where dt has dt_rank dims, B and C have d_state dims
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)

        # dt projection: dt_rank -> d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt bias for stability
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Initialize bias so that softplus(bias) is between 0.001 and 0.1
        dt_bias = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        # Inverse of softplus to get the bias before softplus
        self.dt_proj.bias.data = torch.log(torch.expm1(dt_bias))

        # SSM parameter A (structured, not input-dependent)
        # Initialize as -log(1, 2, ..., d_state) per dimension — HiPPO-style
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))  # store as log for numerical stability

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # 1. Input projection -> x_branch and z_branch
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # 2. Causal 1D convolution on x_branch
        # Conv1d expects (B, C, L)
        x_branch = x_branch.transpose(1, 2)  # (B, d_inner, L)
        x_branch = self.conv1d(x_branch)[:, :, :seq_len]  # truncate for causal
        x_branch = x_branch.transpose(1, 2)  # (B, L, d_inner)

        # 3. SiLU activation
        x_branch = F.silu(x_branch)

        # 4. Compute input-dependent SSM parameters
        x_proj_out = self.x_proj(x_branch)  # (B, L, dt_rank + 2*d_state)
        dt, B, C = x_proj_out.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Project dt to d_inner and apply softplus
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = F.softplus(dt)  # ensure positive
        dt = dt.clamp(min=1e-4, max=10.0)  # clamp for numerical stability

        # Get A (negative for stability)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # 5. Selective SSM scan (sequential, since we're on CPU)
        y = self._selective_scan(x_branch, dt, A, B, C)

        # 6. Skip connection with D
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_branch

        # 7. Gate with z branch
        y = y * F.silu(z)

        # 8. Output projection
        y = self.out_proj(y)

        return y

    def _selective_scan(self, x, dt, A, B, C):
        """
        Selective SSM scan — the core recurrence.

        Discretizes the continuous SSM with input-dependent delta (dt):
            A_bar = exp(dt * A)
            B_bar = dt * B
            h_t = A_bar * h_{t-1} + B_bar * x_t
            y_t = C_t . h_t

        Args:
            x: (B, L, d_inner) — input after conv+SiLU
            dt: (B, L, d_inner) — input-dependent step size (positive)
            A: (d_inner, d_state) — state transition (negative)
            B: (B, L, d_state) — input-dependent input matrix
            C: (B, L, d_state) — input-dependent output matrix

        Returns:
            y: (B, L, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        # Discretize A: A_bar = exp(dt * A)
        # dt: (B, L, d_inner), A: (d_inner, d_state)
        # We compute dt_A = dt.unsqueeze(-1) * A -> (B, L, d_inner, d_state)
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # (B, L, d_inner, d_state)
        dt_A = dt_A.clamp(min=-20.0, max=0.0)  # clamp for numerical stability
        A_bar = torch.exp(dt_A)  # (B, L, d_inner, d_state)

        # Discretize B: B_bar = dt * B (simplified Euler discretization)
        # dt: (B, L, d_inner), B: (B, L, d_state)
        # B_bar_t = dt_t.unsqueeze(-1) * B_t.unsqueeze(-2) -> would be (B,L,d_inner,d_state)
        # But for efficiency, we compute it step by step in the scan

        # Initialize hidden state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)

        # Sequential scan
        outputs = []
        for t in range(seq_len):
            # Current inputs at time t
            x_t = x[:, t, :]          # (B, d_inner)
            dt_t = dt[:, t, :]        # (B, d_inner)
            B_t = B[:, t, :]          # (B, d_state)
            C_t = C[:, t, :]          # (B, d_state)
            A_bar_t = A_bar[:, t]     # (B, d_inner, d_state)

            # B_bar_t = dt_t * B_t — broadcast: (B, d_inner, 1) * (B, 1, d_state)
            B_bar_t = dt_t.unsqueeze(-1) * B_t.unsqueeze(-2)  # (B, d_inner, d_state)

            # State update: h = A_bar * h + B_bar * x
            h = A_bar_t * h + B_bar_t * x_t.unsqueeze(-1)  # (B, d_inner, d_state)

            # Output: y_t = (h * C_t).sum(-1) = dot product over d_state
            y_t = (h * C_t.unsqueeze(-2)).sum(-1)  # (B, d_inner)

            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        return y


class MambaNIDS(nn.Module):
    """
    Mamba-based Network Intrusion Detection System classifier.

    Architecture:
      input_proj -> N x SelectiveSSMBlock -> global average pool -> classifier head
    """

    def __init__(self, input_dim: int, num_classes: int = 10,
                 d_model: int = 64, n_layers: int = 2,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: Number of input features per flow.
            num_classes: Number of attack categories (default 10 for UNSW-NB15).
            d_model: Hidden dimension.
            n_layers: Number of stacked SSM blocks.
            d_state: SSM state dimension.
            d_conv: Causal convolution kernel size.
            expand: Expansion factor for inner dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_model = d_model

        # Project input features to d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Stack of SSM blocks with residual connections and layer norm
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                SelectiveSSMBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
            )
            self.norms.append(nn.LayerNorm(d_model))

        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) — sequence of flow features
        Returns:
            logits: (batch, num_classes)
        """
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)

        # SSM blocks with residual connections
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = norm(layer(x) + residual)

        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch, d_model)

        # Classification
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    print("Testing MambaNIDS...")
    batch_size = 4
    seq_len = 32
    input_dim = 42
    num_classes = 10

    model = MambaNIDS(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=64,
        n_layers=2,
        d_state=16,
    )

    x = torch.randn(batch_size, seq_len, input_dim)
    logits = model(x)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Parameters:   {model.count_parameters():,}")
    assert logits.shape == (batch_size, num_classes), \
        f"Expected ({batch_size}, {num_classes}), got {logits.shape}"
    print("  ✓ Forward pass OK!")
