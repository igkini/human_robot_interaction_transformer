import os
from pathlib import Path
from typing import Dict, List, Optional
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import math


def to_device(batch, device: torch.device):
    """Recursively move tensors (and nested dict/list/tuple) to device."""
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(to_device(x, device) for x in batch)
    elif torch.is_tensor(batch):
        return batch.to(device)
    return batch

class EarlyStopping:
    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.0) -> None:
        self.patience: int = patience
        self.verbose: bool = verbose
        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False
        self.val_loss_min: float = float("inf")
        self.delta: float = delta
    
    def __call__(self, val_loss: float, model: nn.Module, path: str):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving best checkpoint...")
        torch.save(model.state_dict(), os.path.join(path, "checkpoint.pth"))
        self.val_loss_min = val_loss

def compute_metrics(
    preds: torch.Tensor,        # (b,a,t,2)
    target: torch.Tensor,       # (b,a,t,2)
    mask: torch.Tensor,         # (b,a,t,1) with 0/1
) -> Dict[str, torch.Tensor]:

    # Ensure dtypes/shapes
    mask2 = mask.expand_as(preds).to(preds.dtype)  # (b,a,t,2)

    # Per-point diffs
    diff = (preds - target) * mask2
    l1 = diff.abs()
    l2 = diff.pow(2)

    # Counts for normalization
    valid_coord_count = mask2.sum()                         # counts coords (…×2)
    valid_point_count = mask[..., 0].sum()                  # counts points (no ×2)

    # Guard against empty masks (keep differentiable)
    eps = torch.finfo(preds.dtype).eps
    denom_coords = torch.clamp(valid_coord_count, min=eps)
    denom_points = torch.clamp(valid_point_count,  min=eps)

    # Scalar tensor metrics (differentiable)
    mae_t  = l1.sum() / denom_coords
    mse_t  = l2.sum() / denom_coords
    rmse_t = torch.sqrt(mse_t + eps)

    # ADE: L2 distance per point, averaged over valid points
    disp = torch.norm((preds - target) * mask2, dim=-1)     # (b,a,t)
    ade_t = disp.sum() / denom_points

    # FDE: final step only
    final_mask = mask[:, :, -1, 0]                          # (b,a)
    final_pred = preds[:, :, -1, :]
    final_tgt  = target[:, :, -1, :]
    fde_dist   = torch.norm(final_pred - final_tgt, dim=-1) # (b,a)
    denom_final = torch.clamp(final_mask.sum(), min=eps)
    fde_t = (fde_dist * final_mask).sum() / denom_final

    metrics = {"mae": mae_t, "mse": mse_t, "rmse": rmse_t, "ade": ade_t, "fde": fde_t}
    
    return metrics

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    metric: str = 'mae'
) -> float:
    model.train()
    running_loss, n_batches = 0.0, 0
    bar = tqdm(dataloader, total=len(dataloader), desc="Train", leave=False)

    for past_seq, future_seq in bar:
        past_seq   = to_device(past_seq, device)
        future_seq = to_device(future_seq, device)

        preds   = model(past_seq)  # (b,a,t,2)
        metrics = compute_metrics(
            preds,
            future_seq["prediction_pos"],
            future_seq["prediction_pos_mask"],
        )
        loss = metrics[metric]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().cpu())
        running_loss += loss_val
        n_batches += 1
        bar.set_postfix(loss=running_loss / n_batches)

    return running_loss / max(1, n_batches)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    description: str,
    metric: str = 'ade'
) -> float:

    model.eval()
    v_loss, n_batches = 0.0, 0
    bar = tqdm(dataloader, total=len(dataloader), desc=description, leave=False)

    with torch.no_grad():
        for past_seq, future_seq in bar:
            past_seq = to_device(past_seq, device)
            future_seq = to_device(future_seq, device)

            preds = model(past_seq)
            metrics = compute_metrics(preds, future_seq["prediction_pos"], future_seq["prediction_pos_mask"])
            loss = metrics[metric]

            v_loss += loss
            n_batches += 1
            bar.set_postfix(loss=v_loss / n_batches)

    return v_loss / max(1, n_batches) 

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    ckpt_dir: str,
    metric: str = 'ade'
) -> Dict[str, List[float]]:

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.3)
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "test_loss": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, metric=metric)

        # Validate
        val_loss = validate(model, val_loader, device, description="Validation", metric=metric)
        scheduler.step(val_loss)

        # Test after each epoch
        test_loss = validate(model, test_loader, device, description="Test", metric=metric)

        print(f"train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f} | test_loss: {test_loss:.6f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["test_loss"].append(test_loss)

        # Save epoch checkpoint
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
            },
            os.path.join(ckpt_dir, f"model_epoch_{epoch + 1:03d}.pth"),
        )
        print(f"✅ Saved epoch checkpoint to {ckpt_dir}/model_epoch_{epoch + 1:03d}.pth")

        # Early stopping on val loss
        early_stopping(val_loss, model, ckpt_dir)
        if early_stopping.early_stop:
            print(f"⏹️ Early stopping triggered at epoch {epoch + 1}")
            break

        torch.cuda.empty_cache()

    return history