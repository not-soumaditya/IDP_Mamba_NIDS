# Mamba NIDS — Real-Time Cyber Attack Detection using State Space Models

**Innovative Design Project (IDP), Fall Semester 2026-27, VIT Chennai**
> A real-time Network Intrusion Detection System powered by [Mamba](https://arxiv.org/abs/2312.00752), a State Space Model architecture that processes network traffic sequences in **linear time** with **constant memory**.

---

## Problem Statement

Traditional intrusion detection is either **too dumb** (signature-based, can't detect new attacks) or **too slow** (Transformer-based O(L²) cost makes real-time monitoring impractical).

This project builds a **Mamba-based NIDS** that:
- ✅ Detects **unknown attacks** (learned from patterns, not hardcoded rules)
- ✅ Understands **temporal patterns** (DDoS ramp-ups, port scan sequences, C2 beaconing)
- ✅ Runs in **real-time** (~1ms inference per sequence)
- ✅ Uses **constant memory** regardless of traffic history length
- ✅ Is **lightweight** (~165K parameters, ~645 KB model size)

## Performance

| Model | Accuracy | Macro F1 | Temporal Awareness | Parameters |
|-------|----------|----------|--------------------|------------|
| Random Forest | ~93% | ~0.72 | ❌ | N/A |
| XGBoost | ~94% | ~0.76 | ❌ | N/A |
| **Mamba NIDS (Ours)** | **~89%** | **~0.78** | **✅** | **165K** |

> Mamba achieves comparable accuracy while additionally capturing temporal attack patterns that traditional ML cannot detect.


## 📁 Project Structure

```
├── model/
│   ├── mamba_nids.py          # Mamba model (GPU + CPU fallback)
│   ├── dataset.py             # Sliding window sequence dataset
│   ├── checkpoints/           # Saved model weights (.pth)
│   └── __init__.py
├── backend/
│   ├── main.py                # FastAPI server + WebSocket
│   └── requirements.txt
├── frontend/
│   └── index.html             # Dark-mode command center dashboard
├── notebooks/
│   └── kaggle_mamba_nids.py   # Kaggle training notebook (copy-paste ready)
├── .gitignore
└── README.md
```

## Project Instructions

### 1. Train the Model (on Kaggle)

1. Go to [kaggle.com](https://kaggle.com) → **New Notebook**
2. Attach the **UNSW-NB15** dataset
3. Enable **GPU T4** and **Internet**
4. Copy cells from [`notebooks/kaggle_mamba_nids.py`](notebooks/kaggle_mamba_nids.py)
5. Download the saved `mamba_nids_complete.pth` file
6. Place it in `model/checkpoints/`



## Key Concept

### Why Mamba over Transformers?

| | Transformer | Mamba |
|---|---|---|
| Inference memory | O(L) — grows with sequence | **O(1) — constant** |
| Computation | O(L²) — quadratic | **O(L) — linear** |
| 1,000 packets | 1,000,000 operations | **1,000 operations** |
| 10,000 packets | 100,000,000 operations | **10,000 operations** |

### The Selective State Space Equation

```
hₖ = Āₖ · hₖ₋₁ + B̄ₖ · xₖ      (state update)
yₖ = Cₖ · hₖ                    (output)

Where Āₖ, B̄ₖ, Cₖ are INPUT-DEPENDENT (selective)
```

The hidden state `h` is a fixed-size vector (~64 floats) that compresses the entire traffic history. Suspicious inputs (high Δ) dramatically shift the state. Normal inputs barely change it.


## Tech Stack

- **AI**: PyTorch, mamba-ssm (or pure PyTorch fallback)
- **Dataset**: UNSW-NB15 (Kaggle)
- **Baselines**: scikit-learn (Random Forest), XGBoost

## References

1. Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752
2. *NIDS-Mamba: Lightweight Network Intrusion Detection for IoT Sensor Networks via State Space Models.* Sensors, 2026.
3. Moustafa, N., & Slay, J. (2015). *UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection.* MilCIS 2015.

## Team

| Name | Reg. No. |
|------|----------|
| Soumaditya Mukherjee | 25BCE1118 |
| Ramsha Riyasat Fatima | 25BCE1117 |
| Kaivalya Lehekar | 25BCE1431 |

**Department of Computer Science and Engineering (SCOPE)**
Vellore Institute of Technology, Chennai
