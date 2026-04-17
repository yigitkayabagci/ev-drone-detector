"""Training script for SPGNet drone detector.

Usage:
    uv run python scripts/train.py --config configs/default.yaml
    uv run python scripts/train.py --config configs/default.yaml --device cuda:0
    uv run python scripts/train.py --synthetic  # Use synthetic data for testing
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ev_drone_detector.data.dataset import EvUAVDataset, SyntheticEventDataset
from ev_drone_detector.data.event_repr import collate_events, sparse_to_device
from ev_drone_detector.losses.stc_loss import STCLoss
from ev_drone_detector.models.spgnet import SPGNet
from ev_drone_detector.utils.config import load_config
from ev_drone_detector.utils.eval import compute_accuracy, compute_iou


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SPGNet drone detector")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    return parser.parse_args()


def make_collate_fn(spatial_shape: list[int]):
    """Create a collate function with the given spatial shape."""
    def collate_fn(batch):
        return collate_events(batch, spatial_shape=spatial_shape)
    return collate_fn


def train_one_epoch(
    model: SPGNet,
    loader: DataLoader,
    criterion: STCLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        voxel_tensor = sparse_to_device(batch["voxel_tensor"], device)
        labels = batch["labels"].to(device)
        p2v_map = batch["p2v_map"].to(device)

        optimizer.zero_grad()
        predictions, voxel_output = model(voxel_tensor)

        loss = criterion(voxel_output, p2v_map, predictions, labels)
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            pred_np = predictions.squeeze(-1)[p2v_map].cpu().numpy()
            labels_np = labels.cpu().numpy()
            iou = compute_iou(pred_np, labels_np)
            acc = compute_accuracy(pred_np, labels_np)

        total_loss += loss.item()
        total_iou += iou
        total_acc += acc
        n_batches += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            iou=f"{iou:.4f}",
            acc=f"{acc:.4f}",
        )

    return {
        "loss": total_loss / max(n_batches, 1),
        "iou": total_iou / max(n_batches, 1),
        "acc": total_acc / max(n_batches, 1),
    }


@torch.no_grad()
def validate(
    model: SPGNet,
    loader: DataLoader,
    criterion: STCLoss,
    device: torch.device,
) -> dict:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Validating"):
        voxel_tensor = sparse_to_device(batch["voxel_tensor"], device)
        labels = batch["labels"].to(device)
        p2v_map = batch["p2v_map"].to(device)

        predictions, voxel_output = model(voxel_tensor)
        loss = criterion(voxel_output, p2v_map, predictions, labels)

        pred_np = predictions.squeeze(-1)[p2v_map].cpu().numpy()
        labels_np = labels.cpu().numpy()

        total_loss += loss.item()
        total_iou += compute_iou(pred_np, labels_np)
        total_acc += compute_accuracy(pred_np, labels_np)
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "iou": total_iou / max(n_batches, 1),
        "acc": total_acc / max(n_batches, 1),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.training.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Spatial shape (padded, from config)
    spatial_shape = cfg.sensor.spatial_shape

    # Dataset
    if args.synthetic:
        print("Using synthetic dataset for testing")
        train_dataset = SyntheticEventDataset(
            num_samples=50, num_events=5000,
            resolution=cfg.sensor.resolution, whole_t=cfg.sensor.whole_t,
        )
        val_dataset = SyntheticEventDataset(
            num_samples=10, num_events=5000,
            resolution=cfg.sensor.resolution, whole_t=cfg.sensor.whole_t,
        )
    else:
        train_dataset = EvUAVDataset(
            cfg.data.train_dir, max_events=cfg.data.max_events, split="train"
        )
        val_dataset = EvUAVDataset(
            cfg.data.val_dir, max_events=cfg.data.max_events, split="val"
        )

    collate_fn = make_collate_fn(spatial_shape)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Model
    model = SPGNet(
        input_channel=cfg.model.input_channel,
        width=cfg.model.width,
        spatial_shape=spatial_shape,
        dilations=cfg.model.dilations,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    # Loss, optimizer, scheduler
    criterion = STCLoss(
        k=cfg.loss.stc.k, tau=cfg.loss.stc.tau,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.training.scheduler.step_size,
        gamma=cfg.training.scheduler.gamma,
    )

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    save_dir = Path(cfg.training.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_iou = 0.0
    best_loss = float("inf")

    for epoch in range(start_epoch, cfg.training.epochs):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        scheduler.step()

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"train_iou={train_metrics['iou']:.4f}, "
            f"train_acc={train_metrics['acc']:.4f}, "
            f"lr={scheduler.get_last_lr()[0]:.6f}"
        )

        # Validate
        if epoch >= cfg.training.val_start_epoch:
            val_metrics = validate(model, val_loader, criterion, device)
            print(
                f"  val_loss={val_metrics['loss']:.4f}, "
                f"val_iou={val_metrics['iou']:.4f}, "
                f"val_acc={val_metrics['acc']:.4f}"
            )

            # Save best models
            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }

            if val_metrics["iou"] > best_iou:
                best_iou = val_metrics["iou"]
                torch.save(ckpt_data, save_dir / "best_iou.pt")
                print(f"  Saved best IoU model: {best_iou:.4f}")

            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                torch.save(ckpt_data, save_dir / "best_loss.pt")
                print(f"  Saved best loss model: {best_loss:.4f}")

    print(f"\nTraining complete. Best IoU: {best_iou:.4f}, Best Loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
