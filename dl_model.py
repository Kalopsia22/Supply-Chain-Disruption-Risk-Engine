"""
Deep Learning model (PyTorch): a feedforward neural network with learned
categorical embeddings, trained as a genuine alternative to the XGBoost
classifier — same leading-indicator features, same train/test split, so
results are directly comparable.

Architecture: each categorical feature (Origin, Destination, Route Type,
Mode, Category) gets its own learned embedding; these are concatenated with
the standardized numeric features and passed through a 3-layer MLP with
dropout and batch norm, ending in a sigmoid output (delay probability).
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score

from data_utils import (
    load_scored_data, FEATURES_NUMERIC, FEATURES_CATEGORICAL, _build_input_row,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "deep_risk_net.pt")
META_PATH = os.path.join(BASE_DIR, "deep_risk_net_meta.json")

DEVICE = torch.device("cpu")


class DeepRiskNet(nn.Module):
    """Embeddings for categoricals + dense layers for numerics -> delay probability."""

    def __init__(self, cat_cardinalities: dict, n_numeric: int, emb_dim: int = 8):
        super().__init__()
        self.cat_cols = list(cat_cardinalities.keys())
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(card + 1, min(emb_dim, (card + 1) // 2 + 1))  # +1 for unseen/unknown index
            for col, card in cat_cardinalities.items()
        })
        total_emb_dim = sum(e.embedding_dim for e in self.embeddings.values())

        self.net = nn.Sequential(
            nn.Linear(total_emb_dim + n_numeric, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1),
        )

    def forward(self, x_num, x_cat):
        embs = [self.embeddings[col](x_cat[:, i]) for i, col in enumerate(self.cat_cols)]
        x = torch.cat(embs + [x_num], dim=1)
        return self.net(x).squeeze(-1)


def _encode_categoricals(df: pd.DataFrame, encoders: dict = None, fit: bool = False):
    """Label-encode categoricals with a reserved 'unknown' index for unseen values at inference."""
    encoded = np.zeros((len(df), len(FEATURES_CATEGORICAL)), dtype=np.int64)
    if fit:
        encoders = {}
    for i, col in enumerate(FEATURES_CATEGORICAL):
        if fit:
            le = LabelEncoder()
            encoded[:, i] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            unknown_idx = len(le.classes_)  # reserved slot for unseen categories at inference
            mapping = {c: idx for idx, c in enumerate(le.classes_)}
            encoded[:, i] = df[col].astype(str).map(lambda v: mapping.get(v, unknown_idx)).values
    return encoded, encoders


def train_deep_model(epochs: int = 60, batch_size: int = 128, lr: float = 1e-3, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    df = load_scored_data()
    X = df[FEATURES_NUMERIC + FEATURES_CATEGORICAL].copy()
    y = df["Is_Delayed"].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[FEATURES_NUMERIC]).astype(np.float32)
    X_test_num = scaler.transform(X_test[FEATURES_NUMERIC]).astype(np.float32)

    X_train_cat, encoders = _encode_categoricals(X_train, fit=True)
    X_test_cat, _ = _encode_categoricals(X_test, encoders=encoders, fit=False)

    cat_cardinalities = {col: len(encoders[col].classes_) for col in FEATURES_CATEGORICAL}
    model = DeepRiskNet(cat_cardinalities, n_numeric=len(FEATURES_NUMERIC)).to(DEVICE)

    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    X_train_num_t = torch.tensor(X_train_num)
    X_train_cat_t = torch.tensor(X_train_cat, dtype=torch.long)
    y_train_t = torch.tensor(y_train)

    n = len(X_train_num_t)
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(X_train_num_t[idx], X_train_cat_t[idx])
            loss = criterion(logits, y_train_t[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - loss: {total_loss/n:.4f}")

    model.eval()
    with torch.no_grad():
        test_logits = model(torch.tensor(X_test_num), torch.tensor(X_test_cat, dtype=torch.long))
        test_proba = torch.sigmoid(test_logits).numpy()

    auc = roc_auc_score(y_test, test_proba)
    ap = average_precision_score(y_test, test_proba)
    print(f"\n=== Deep Learning Classifier (PyTorch) ===")
    print(f"ROC-AUC: {auc:.3f}  |  Avg Precision (PR-AUC): {ap:.3f}")

    torch.save(model.state_dict(), MODEL_PATH)
    meta = {
        "cat_cardinalities": cat_cardinalities,
        "categories": {col: list(encoders[col].classes_) for col in FEATURES_CATEGORICAL},
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "numeric_features": FEATURES_NUMERIC,
        "categorical_features": FEATURES_CATEGORICAL,
        "test_auc": auc,
        "test_ap": ap,
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    return model, meta


_model_cache = None
_meta_cache = None


def load_deep_model():
    global _model_cache, _meta_cache
    if _model_cache is not None:
        return _model_cache, _meta_cache

    if not os.path.exists(META_PATH) or not os.path.exists(MODEL_PATH):
        present = os.listdir(BASE_DIR)
        raise FileNotFoundError(
            f"Deep learning model files not found in {BASE_DIR}. Expected "
            f"'{os.path.basename(MODEL_PATH)}' and '{os.path.basename(META_PATH)}'.\n"
            f"Files actually present: {present}\n"
            "These are generated by running `python dl_model.py` once, or they need to be "
            "committed to the repo (check .gitignore isn't excluding *.pt or *.json files) "
            "and pushed alongside app.py."
        )

    with open(META_PATH) as f:
        meta = json.load(f)

    model = DeepRiskNet(meta["cat_cardinalities"], n_numeric=len(meta["numeric_features"]))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    _model_cache, _meta_cache = model, meta
    return model, meta


def score_single_shipment_dl(inputs: dict) -> dict:
    """Score a single hypothetical shipment with the PyTorch deep learning model."""
    model, meta = load_deep_model()
    row = _build_input_row(inputs)

    num_vals = row[meta["numeric_features"]].values.astype(np.float32)
    mean = np.array(meta["scaler_mean"], dtype=np.float32)
    scale = np.array(meta["scaler_scale"], dtype=np.float32)
    num_scaled = (num_vals - mean) / scale

    cat_vals = np.zeros((1, len(meta["categorical_features"])), dtype=np.int64)
    for i, col in enumerate(meta["categorical_features"]):
        classes = meta["categories"][col]
        val = str(row[col].iloc[0])
        cat_vals[0, i] = classes.index(val) if val in classes else len(classes)

    with torch.no_grad():
        logit = model(torch.tensor(num_scaled), torch.tensor(cat_vals, dtype=torch.long))
        proba = torch.sigmoid(logit).item()

    return {"probability": float(proba)}


if __name__ == "__main__":
    train_deep_model()
