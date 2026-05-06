from __future__ import annotations

import argparse
from pathlib import Path

import config_ai
from dataset_ai import CoilSegDataset, build_eval_transform, build_train_transform, collect_sample_pairs
from io_utils_ai import ensure_dir
from segment_model import build_unetplusplus_model, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="U-Net++ EfficientNet-B4 기반 coil segmentation 학습")
    parser.add_argument("--train-images-dir", default=str(config_ai.TRAIN_IMAGES_DIR))
    parser.add_argument("--train-masks-dir", default=str(config_ai.TRAIN_MASKS_DIR))
    parser.add_argument("--val-images-dir", default=str(config_ai.VAL_IMAGES_DIR))
    parser.add_argument("--val-masks-dir", default=str(config_ai.VAL_MASKS_DIR))
    parser.add_argument("--checkpoint-path", default=str(config_ai.BEST_CHECKPOINT_PATH))
    parser.add_argument("--last-checkpoint-path", default=str(config_ai.LAST_CHECKPOINT_PATH))
    parser.add_argument("--input-size", type=int, default=config_ai.INPUT_SIZE)
    parser.add_argument("--batch-size", type=int, default=config_ai.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config_ai.EPOCHS)
    parser.add_argument("--lr", type=float, default=config_ai.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=config_ai.WEIGHT_DECAY)
    parser.add_argument("--device", default=config_ai.DEVICE)
    parser.add_argument("--num-workers", type=int, default=config_ai.NUM_WORKERS)
    parser.add_argument("--encoder-name", default=config_ai.ENCODER_NAME)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume-from", default="")
    return parser.parse_args()


def dice_loss(logits, targets, eps: float = 1e-6):
    import torch

    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


class BCEDiceLoss:
    def __init__(self) -> None:
        import torch.nn as nn

        self.bce = nn.BCEWithLogitsLoss()

    def __call__(self, logits, targets):
        return self.bce(logits, targets) + dice_loss(logits, targets)


# boundary-aware loss를 나중에 추가할 경우 이 함수에서 확장한다.
def build_loss():
    return BCEDiceLoss()


def compute_batch_dice(logits, targets, eps: float = 1e-6) -> float:
    import torch

    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return float(dice.mean().detach().cpu().item())


def run_epoch(model, loader, criterion, optimizer, device, scaler, *, train: bool, use_amp: bool):
    import torch

    model.train(train)
    total_loss = 0.0
    total_dice = 0.0
    total_steps = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            autocast_enabled = use_amp and device.type == "cuda"
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                logits = model(images)
                loss = criterion(logits, masks)

            if train:
                if scaler is not None and autocast_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        total_dice += compute_batch_dice(logits, masks)
        total_steps += 1

    if total_steps == 0:
        return 0.0, 0.0
    return total_loss / total_steps, total_dice / total_steps


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader

    train_samples = collect_sample_pairs(Path(args.train_images_dir), Path(args.train_masks_dir))
    val_samples = collect_sample_pairs(Path(args.val_images_dir), Path(args.val_masks_dir))

    train_dataset = CoilSegDataset(
        train_samples,
        transform=build_train_transform(args.input_size),
        input_size=args.input_size,
    )
    val_dataset = CoilSegDataset(
        val_samples,
        transform=build_eval_transform(args.input_size),
        input_size=args.input_size,
    )

    pin_memory = config_ai.PIN_MEMORY and args.device in {"auto", "cuda"}
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model, resolved_encoder = build_unetplusplus_model(
        encoder_name=args.encoder_name,
        encoder_weights="imagenet",
    )
    device = resolve_device(args.device)
    model.to(device)

    criterion = build_loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(not args.no_amp and device.type == "cuda"))

    checkpoint_path = Path(args.checkpoint_path)
    last_checkpoint_path = Path(args.last_checkpoint_path)
    ensure_dir(checkpoint_path.parent)
    ensure_dir(last_checkpoint_path.parent)

    best_val_dice = -1.0
    start_epoch = 1

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint가 없습니다: {resume_path}")

        checkpoint = torch.load(str(resume_path), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler_state_dict = checkpoint.get("scaler_state_dict")
        if scaler_state_dict is not None and scaler is not None:
            scaler.load_state_dict(scaler_state_dict)

        best_val_dice = float(checkpoint.get("best_val_dice", -1.0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(
            f"resume 완료\n"
            f"- checkpoint: {resume_path.resolve()}\n"
            f"- 다음 epoch: {start_epoch}\n"
            f"- best_val_dice: {best_val_dice:.4f}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_dice = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            train=True,
            use_amp=(not args.no_amp),
        )
        val_loss, val_dice = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            scaler,
            train=False,
            use_amp=False,
        )

        checkpoint = {
            "epoch": epoch,
            "encoder_name": resolved_encoder,
            "architecture": config_ai.ARCHITECTURE,
            "input_size": args.input_size,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_val_dice": max(best_val_dice, val_dice),
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_dice": val_dice,
        }
        torch.save(checkpoint, str(last_checkpoint_path))
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(checkpoint, str(checkpoint_path))

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
            f"best_val_dice={best_val_dice:.4f}"
        )

    print(
        f"\n학습 완료\n"
        f"- best checkpoint: {checkpoint_path.resolve()}\n"
        f"- last checkpoint: {last_checkpoint_path.resolve()}"
    )


if __name__ == "__main__":
    main()
