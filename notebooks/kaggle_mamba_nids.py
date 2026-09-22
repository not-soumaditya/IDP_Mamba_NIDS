"""
Mamba NIDS — Kaggle Training Notebook
=====================================
Copy this into a Kaggle Notebook cell-by-cell.
See docs/kaggle_guide.md for detailed instructions.

Setup:
  1. Attach UNSW-NB15 dataset (by mrwellsdavid)
  2. Enable GPU T4 in session options
  3. Enable Internet access
"""

# %% [markdown]
# # Mamba NIDS — Training Notebook

# %% Cell 1: Install dependencies
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"  # Tesla T4
os.environ["MAX_JOBS"] = "2"

# !pip install ninja packaging -q
# !pip install causal-conv1d --no-build-isolation --no-cache-dir 2>&1 | tail -5
# !pip install mamba-ssm --no-build-isolation --no-cache-dir 2>&1 | tail -5
# !pip install xgboost -q

# %% Cell 2: Imports
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Device: {device}")

# %% Cell 3: Load dataset
input_dir = '/kaggle/input'
for dirname, _, filenames in os.walk(input_dir):
    for filename in filenames:
        filepath = os.path.join(dirname, filename)
        size_mb = os.path.getsize(filepath) / 1e6
        print(f"  {filepath}  ({size_mb:.1f} MB)")

# %% Cell 4: Read CSVs
# UPDATE THESE PATHS if needed
TRAIN_PATH = '/kaggle/input/unsw-nb15/UNSW_NB15_training-set.csv'
TEST_PATH  = '/kaggle/input/unsw-nb15/UNSW_NB15_testing-set.csv'

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# %% Cell 5: Attack distribution
train_df['attack_cat'] = train_df['attack_cat'].fillna('Normal').str.strip()
test_df['attack_cat']  = test_df['attack_cat'].fillna('Normal').str.strip()
train_df.loc[train_df['label'] == 0, 'attack_cat'] = 'Normal'
test_df.loc[test_df['label'] == 0, 'attack_cat'] = 'Normal'

print(train_df['attack_cat'].value_counts())

# %% Cell 6: Feature engineering
train_df = train_df.drop(columns=['id'], errors='ignore')
test_df  = test_df.drop(columns=['id'], errors='ignore')

exclude_cols = ['label', 'attack_cat']
feature_cols = [c for c in train_df.columns if c not in exclude_cols]
categorical_cols = train_df[feature_cols].select_dtypes(include=['object']).columns.tolist()
numerical_cols = [c for c in feature_cols if c not in categorical_cols]

train_df[feature_cols] = train_df[feature_cols].fillna(0)
test_df[feature_cols]  = test_df[feature_cols].fillna(0)

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str)])
    le.fit(combined)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col]  = le.transform(test_df[col].astype(str))
    label_encoders[col] = le

scaler = StandardScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
test_df[numerical_cols]  = scaler.transform(test_df[numerical_cols])

attack_encoder = LabelEncoder()
train_df['attack_label'] = attack_encoder.fit_transform(train_df['attack_cat'])
test_df['attack_label']  = attack_encoder.transform(test_df['attack_cat'])
num_classes = len(attack_encoder.classes_)
print(f"Features: {len(feature_cols)}, Classes: {num_classes}")
print(f"Classes: {list(attack_encoder.classes_)}")

# %% Cell 7: Sequence dataset
class NIDSSequenceDataset(Dataset):
    def __init__(self, dataframe, feature_cols, label_col, seq_len=32, stride=1):
        self.seq_len = seq_len
        self.features = dataframe[feature_cols].values.astype(np.float32)
        self.labels = dataframe[label_col].values.astype(np.int64)
        self.valid_indices = list(range(0, len(self.features) - seq_len + 1, stride))

    def __len__(self): return len(self.valid_indices)

    def __getitem__(self, idx):
        start = self.valid_indices[idx]
        end = start + self.seq_len
        return torch.tensor(self.features[start:end]), torch.tensor(self.labels[end - 1])

SEQ_LEN, BATCH_SIZE, STRIDE = 32, 64, 4
train_dataset = NIDSSequenceDataset(train_df, feature_cols, 'attack_label', SEQ_LEN, STRIDE)
test_dataset  = NIDSSequenceDataset(test_df, feature_cols, 'attack_label', SEQ_LEN, STRIDE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print(f"Train sequences: {len(train_dataset):,}, Test sequences: {len(test_dataset):,}")

# %% Cell 8: Model definition
USING_MAMBA_SSM = False
try:
    from mamba_ssm import Mamba
    USING_MAMBA_SSM = True
    print("Using mamba-ssm (GPU-accelerated)")
except ImportError:
    print("Using pure PyTorch SSM fallback")

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = d_model * expand
        self.d_inner, self.d_state = d_inner, d_state
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv-1, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)
        A = torch.arange(1, d_state+1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        b, l, _ = x.shape
        xz = self.in_proj(x)
        x_b, z = xz.chunk(2, dim=-1)
        x_b = torch.nn.functional.silu(self.conv1d(x_b.transpose(1,2))[:,:,:l].transpose(1,2))
        xp = self.x_proj(x_b)
        B, C = xp[:,:,:self.d_state], xp[:,:,self.d_state:2*self.d_state]
        delta = self.dt_proj(torch.nn.functional.softplus(xp[:,:,-1:]))
        A = -torch.exp(self.A_log)
        h = torch.zeros(b, self.d_inner, self.d_state, device=x.device)
        outs = []
        for t in range(l):
            dt = delta[:,t,:]
            h = torch.exp(dt.unsqueeze(-1)*A.unsqueeze(0))*h + (dt.unsqueeze(-1)*B[:,t,:].unsqueeze(1))*x_b[:,t,:].unsqueeze(-1)
            outs.append((h*C[:,t,:].unsqueeze(1)).sum(-1) + self.D*x_b[:,t,:])
        return self.out_proj(torch.stack(outs,1) * torch.nn.functional.silu(z))

class MambaNIDS(nn.Module):
    def __init__(self, input_dim, d_model=64, d_state=16, d_conv=4, expand=2, n_layers=4, num_classes=10, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            if USING_MAMBA_SSM:
                self.layers.append(Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand))
            else:
                self.layers.append(SelectiveSSM(d_model, d_state, d_conv, expand))
            self.norms.append(nn.LayerNorm(d_model))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x):
        x = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            x = layer(norm(x)) + x
        return self.classifier(self.dropout(x.mean(1)))

INPUT_DIM = len(feature_cols)
model = MambaNIDS(INPUT_DIM, d_model=64, d_state=16, n_layers=4, num_classes=num_classes).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,} | Size: {total_params*4/1024:.1f} KB")

# %% Cell 9: Class weights
class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=train_df['attack_label'].values)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
print("Class weights:", {c: f"{w:.2f}" for c, w in zip(attack_encoder.classes_, class_weights)})

# %% Cell 10: Training
LR, EPOCHS = 1e-3, 30
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item()
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
    return loss_sum/len(loader), 100.*correct/total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    preds, labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item()
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
        preds.extend(pred.cpu().numpy())
        labels.extend(y.cpu().numpy())
    return loss_sum/len(loader), 100.*correct/total, preds, labels

best_acc = 0
for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
    te_loss, te_acc, preds, labels = evaluate(model, test_loader, criterion)
    scheduler.step()
    marker = ""
    if te_acc > best_acc:
        best_acc = te_acc
        torch.save(model.state_dict(), 'best_mamba_nids.pth')
        marker = " *"
    print(f"E{epoch:02d} | TrL:{tr_loss:.4f} TrA:{tr_acc:.1f}% | TeL:{te_loss:.4f} TeA:{te_acc:.1f}% | {time.time()-t0:.1f}s{marker}")

print(f"\nBest accuracy: {best_acc:.2f}%")

# %% Cell 11: Evaluation
model.load_state_dict(torch.load('best_mamba_nids.pth'))
_, test_acc, preds, labels = evaluate(model, test_loader, criterion)

print(classification_report(labels, preds, target_names=attack_encoder.classes_, digits=4))
macro_f1 = f1_score(labels, preds, average='macro')
weighted_f1 = f1_score(labels, preds, average='weighted')
print(f"Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f}")

cm = confusion_matrix(labels, preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=attack_encoder.classes_, yticklabels=attack_encoder.classes_)
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.title('Mamba NIDS — Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# %% Cell 12: Baselines
X_tr = train_df[feature_cols].values
y_tr = train_df['attack_label'].values
X_te = test_df[feature_cols].values
y_te = test_df['attack_label'].values

t0 = time.time()
rf = RandomForestClassifier(100, max_depth=20, n_jobs=-1, class_weight='balanced', random_state=42)
rf.fit(X_tr, y_tr)
rf_time = time.time()-t0
rf_preds = rf.predict(X_te)
rf_acc = (rf_preds==y_te).mean()*100
rf_f1 = f1_score(y_te, rf_preds, average='macro')
print(f"RF:  Acc={rf_acc:.2f}% F1={rf_f1:.4f} Time={rf_time:.1f}s")

from xgboost import XGBClassifier
t0 = time.time()
xgb = XGBClassifier(200, max_depth=8, learning_rate=0.1, tree_method='hist',
                     eval_metric='mlogloss', random_state=42, use_label_encoder=False)
xgb.fit(X_tr, y_tr, verbose=False)
xgb_time = time.time()-t0
xgb_preds = xgb.predict(X_te)
xgb_acc = (xgb_preds==y_te).mean()*100
xgb_f1 = f1_score(y_te, xgb_preds, average='macro')
print(f"XGB: Acc={xgb_acc:.2f}% F1={xgb_f1:.4f} Time={xgb_time:.1f}s")

# %% Cell 13: Save everything
output_dir = '/kaggle/working'
torch.save({
    'model_state_dict': model.state_dict(),
    'attack_encoder_classes': attack_encoder.classes_.tolist(),
    'feature_cols': feature_cols,
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'hyperparams': {'input_dim': INPUT_DIM, 'd_model': 64, 'd_state': 16, 'n_layers': 4, 'num_classes': num_classes, 'seq_len': SEQ_LEN},
}, os.path.join(output_dir, 'mamba_nids_complete.pth'))
print(f"Saved to {output_dir}/mamba_nids_complete.pth")
