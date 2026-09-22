# Real-Time Cyber Attack Detection using Mamba (State Space Models)

## Project Overview

A Network Intrusion Detection System (NIDS) that classifies network flows as
Normal or one of 9 attack categories using a **Mamba** (Selective State Space
Model) architecture, trained on the **UNSW-NB15** dataset.

### Attack Categories (10 classes)
Normal, Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Research & Data Preparation |
| Phase 2 | 🔄 In Progress | Model Development (CPU prototype done, GPU training pending) |
| Phase 3 | ⬜ Planned | Backend API Integration |
| Phase 4 | ⬜ Planned | Dashboard & Deployment |

## Quick Start

### Prerequisites
- Python 3.10+
- Kaggle API credentials (for dataset download)

### Local Setup (CPU)

```bash
# 1. Clone and navigate to the project
cd IDP_NIDS-Mamba

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the UNSW-NB15 dataset
python data/download_data.py

# 4. Clean and preprocess data
python model/dataset.py

# 5. Train the baseline model (RandomForest)
python model/baselines.py

# 6. Train the CPU Mamba prototype
python model/train_prototype.py

# 7. Run sanity tests
pytest tests/test_sanity.py -v
```

### GPU Training (Kaggle/Colab)

For full Mamba training with the `mamba-ssm` library:

1. Upload `colab_gpu_training.ipynb` to [Kaggle Notebooks](https://www.kaggle.com/code) or [Google Colab](https://colab.research.google.com/)
2. Enable GPU runtime (Settings → Accelerator → GPU)
3. Run all cells — the notebook is self-contained

## Repository Structure

```
IDP_NIDS-Mamba/
├── data/
│   ├── raw/                          # Downloaded CSVs (gitignored)
│   ├── processed/                     # Cleaned/encoded data
│   └── download_data.py              # Kaggle dataset download
├── model/
│   ├── mamba_minimal.py              # Pure-PyTorch selective SSM block
│   ├── dataset.py                    # Data cleaning + sequence dataset
│   ├── baselines.py                  # RandomForest baseline
│   ├── train_prototype.py            # CPU Mamba prototype training
│   ├── evaluate.py                   # Shared evaluation utilities
│   └── checkpoints/                  # Saved model weights (gitignored)
├── backend/
│   └── main.py                       # FastAPI placeholder (/health only)
├── frontend/
│   └── index.html                    # Dashboard placeholder
├── notebooks/
│   └── 01_data_exploration.ipynb     # EDA notebook
├── docs/
│   ├── baseline_results.md           # RandomForest results
│   ├── prototype_results.md          # CPU Mamba results
│   ├── class_distribution.md         # Dataset class distribution
│   └── *.png                         # Charts and plots
├── tests/
│   └── test_sanity.py                # Pytest sanity tests
├── colab_gpu_training.ipynb          # Self-contained GPU training notebook
├── requirements.txt
├── README.md
└── .gitignore
```

## Models

### Baseline: RandomForest
- Trained on flat (non-sequential) features
- Results: see [docs/baseline_results.md](docs/baseline_results.md)

### CPU Prototype: MambaNIDS
- Pure-PyTorch selective SSM implementation (no GPU required)
- Trained on a small data subset to verify forward/backward pass
- Results: see [docs/prototype_results.md](docs/prototype_results.md)

### Full Model: MambaNIDS (GPU)
- Uses `mamba-ssm` library for optimized CUDA kernels
- Trained on full dataset via Kaggle/Colab
- Notebook: [colab_gpu_training.ipynb](colab_gpu_training.ipynb)

## Architecture

```
Input (flow features) → Input Projection → [N × Selective SSM Blocks] → Global Pool → Classifier
                                                    ↑
                                          Causal Conv1D + SiLU
                                          Input-dependent Δ, B, C
                                          Selective State Space Scan
                                          Gated Output Projection
```

## Tech Stack
- **ML Framework:** PyTorch
- **SSM Library:** mamba-ssm (GPU) / custom pure-PyTorch (CPU)
- **Data:** UNSW-NB15 dataset
- **Backend:** FastAPI (Phase 3)
- **Frontend:** Dark-mode dashboard (Phase 4)
- **Evaluation:** scikit-learn metrics, matplotlib visualizations

## Team
VIT 2026-27 — Interdisciplinary Project

## License
Academic use only.
