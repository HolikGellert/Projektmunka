import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from prediction.config import PredictionConfig as Config

logger = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    """Simple tensor dataset for sequence regression."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = Config.BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int | None = None, shuffle: bool = False) -> DataLoader:
    """Utility to create a DataLoader for any split (e.g., test)."""
    ds = SequenceDataset(X, y)
    return DataLoader(ds, batch_size=batch_size or Config.BATCH_SIZE, shuffle=shuffle)


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = Config.EPOCHS,
    lr: float = Config.LEARNING_RATE,
    weight_decay: float = Config.WEIGHT_DECAY,
    grad_clip: float | None = Config.GRAD_CLIP,
    max_batches: int | None = None,
    early_stop_patience: int | None = Config.EARLY_STOP_PATIENCE,
    device: torch.device | str | None = None,
) -> Dict[str, List[float]]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history = {"train_loss": [], "val_loss": []}
    best_state = None
    best_val = float("inf")
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_seen = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader, start=1):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            batch_size = len(X_batch)
            train_loss += loss.item() * batch_size
            train_seen += batch_size

            if max_batches and batch_idx >= max_batches:
                break

        train_loss /= max(train_seen, 1)

        model.eval()
        val_loss = 0.0
        val_seen = 0
        with torch.no_grad():
            for batch_idx, (X_batch, y_batch) in enumerate(val_loader, start=1):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                batch_size = len(X_batch)
                val_loss += loss.item() * batch_size
                val_seen += batch_size

                if max_batches and batch_idx >= max_batches:
                    break

        val_loss /= max(val_seen, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        logger.info("Epoch %d/%d - train_loss=%.4f val_loss=%.4f", epoch, epochs, train_loss, val_loss)
        print(f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        scheduler.step(val_loss)

        # Track best model for early stopping
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            patience_ctr = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if early_stop_patience and patience_ctr >= early_stop_patience:
                print(f"Early stopping at epoch {epoch} (best val_loss={best_val:.4f})")
                break

    # Restore best weights if captured
    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def evaluate_mae(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | str | None = None,
    max_batches: int | None = Config.MAX_EVAL_BATCHES,
) -> float:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()
    absolute_errors = []
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(loader, start=1):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            absolute_errors.append(torch.abs(preds - y_batch).cpu().numpy())
            if max_batches and batch_idx >= max_batches:
                break
    return float(np.concatenate(absolute_errors).mean())


def save_model(model: torch.nn.Module, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Ensure tensors are on CPU to avoid device-specific issues during load/save
        state = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model.state_dict().items()}
        torch.save(state, path)
        logger.info("Saved model to %s", path)
        return path
    except Exception:
        logger.exception("Failed to save model to %s", path)
        raise


def save_metadata(metadata: Dict, path: Path = Config.METADATA_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to %s", path)


def save_scaler(scaler, path: Path = Config.SCALER_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    logger.info("Saved scaler to %s", path)
