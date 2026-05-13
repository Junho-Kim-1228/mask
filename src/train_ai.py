from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--train-metadata", default="")
    parser.add_argument("--oversample-source", action="append", dest="oversample_sources")
    parser.add_argument("--oversample-factor", type=float, default=3.0)
    parser.add_argument("--reset-optimizer-on-resume", action="store_true")
    parser.add_argument("--tversky-alpha", type=float, default=0.30)
    parser.add_argument("--tversky-beta", type=float, default=0.70)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--boundary-weight", type=float, default=3.0)
    parser.add_argument("--val-metadata", default="")
    parser.add_argument("--focus-val-source", action="append", dest="focus_val_sources")
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument(
        "--best-metric",
        choices=("auto", "val_dice", "val_tversky", "focus_val_dice", "focus_val_tversky"),
        default="auto",
    )
    return parser.parse_args()


def compute_tversky_tensor(logits, targets, alpha: float = 0.30, beta: float = 0.70, eps: float = 1e-6):
    import torch

    probs = torch.sigmoid(logits)
    true_pos = (probs * targets).sum(dim=(1, 2, 3))
    false_pos = (probs * (1.0 - targets)).sum(dim=(1, 2, 3))
    false_neg = ((1.0 - probs) * targets).sum(dim=(1, 2, 3))
    return (true_pos + eps) / (true_pos + alpha * false_pos + beta * false_neg + eps)


def build_boundary_weight_map(targets, boundary_weight: float):
    import torch
    import torch.nn.functional as F

    if boundary_weight <= 0:
        return torch.ones_like(targets)

    dilated = F.max_pool2d(targets, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-targets, kernel_size=3, stride=1, padding=1)
    boundary = (dilated - eroded) > 0
    return 1.0 + boundary.float() * boundary_weight


class BoundaryWeightedBCEFocalTverskyLoss:
    def __init__(self, *, alpha: float, beta: float, gamma: float, boundary_weight: float) -> None:
        import torch.nn as nn

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.boundary_weight = boundary_weight

    def __call__(self, logits, targets):
        import torch

        weight_map = build_boundary_weight_map(targets, self.boundary_weight)
        bce_map = self.bce(logits, targets)
        weighted_bce = (bce_map * weight_map).mean()
        tversky = compute_tversky_tensor(
            logits,
            targets,
            alpha=self.alpha,
            beta=self.beta,
        )
        focal_tversky = torch.pow(1.0 - tversky, self.gamma).mean()
        return weighted_bce + focal_tversky


def build_loss(args: argparse.Namespace):
    return BoundaryWeightedBCEFocalTverskyLoss(
        alpha=args.tversky_alpha,
        beta=args.tversky_beta,
        gamma=args.focal_gamma,
        boundary_weight=args.boundary_weight,
    )


def compute_batch_dice(logits, targets, eps: float = 1e-6) -> float:
    import torch

    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return float(dice.mean().detach().cpu().item())


def compute_batch_tversky(
    logits,
    targets,
    alpha: float = 0.30,
    beta: float = 0.70,
    eps: float = 1e-6,
) -> float:
    import torch

    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    true_pos = (preds * targets).sum(dim=(1, 2, 3))
    false_pos = (preds * (1.0 - targets)).sum(dim=(1, 2, 3))
    false_neg = ((1.0 - preds) * targets).sum(dim=(1, 2, 3))
    tversky = (true_pos + eps) / (true_pos + alpha * false_pos + beta * false_neg + eps)
    return float(tversky.mean().detach().cpu().item())


def compute_sample_dice(logits, targets, eps: float = 1e-6):
    import torch

    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2.0 * intersection + eps) / (union + eps)).detach().cpu().tolist()


def compute_sample_tversky(
    logits,
    targets,
    alpha: float = 0.30,
    beta: float = 0.70,
    eps: float = 1e-6,
):
    import torch

    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    true_pos = (preds * targets).sum(dim=(1, 2, 3))
    false_pos = (preds * (1.0 - targets)).sum(dim=(1, 2, 3))
    false_neg = ((1.0 - preds) * targets).sum(dim=(1, 2, 3))
    tversky = (true_pos + eps) / (true_pos + alpha * false_pos + beta * false_neg + eps)
    return tversky.detach().cpu().tolist()


def resolve_train_metadata_path(args: argparse.Namespace) -> Path | None:
    if args.train_metadata:
        path = Path(args.train_metadata)
        return path if path.exists() else None

    train_images_dir = Path(args.train_images_dir)
    candidate = train_images_dir.parent.parent / "train_metadata.csv"
    return candidate if candidate.exists() else None


def load_train_metadata(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}

    metadata: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            stem = (row.get("stem") or "").strip()
            source_dir = (row.get("source_dir") or "").strip()
            if stem:
                metadata[stem] = source_dir
    return metadata


def resolve_val_metadata_path(args: argparse.Namespace) -> Path | None:
    if args.val_metadata:
        path = Path(args.val_metadata)
        return path if path.exists() else None

    val_images_dir = Path(args.val_images_dir)
    candidate = val_images_dir.parent.parent / "val_metadata.csv"
    return candidate if candidate.exists() else None


def build_periodic_checkpoint_path(checkpoint_path: Path, epoch: int) -> Path:
    stem = checkpoint_path.stem
    if stem.endswith("_best"):
        stem = stem[:-5]
    return checkpoint_path.with_name(f"{stem}_epoch{epoch:03d}{checkpoint_path.suffix}")


def resolve_best_metric_name(
    args: argparse.Namespace,
    *,
    focus_sources: set[str],
) -> str:
    if args.best_metric != "auto":
        return args.best_metric
    return "focus_val_tversky" if focus_sources else "val_tversky"


def build_train_sampler(args: argparse.Namespace, samples, metadata: dict[str, str]):
    import torch
    from torch.utils.data import WeightedRandomSampler

    oversample_sources = {s.strip() for s in (args.oversample_sources or []) if s.strip()}
    if not oversample_sources:
        return None, []

    weights: list[float] = []
    boosted_stems: list[str] = []
    for sample in samples:
        stem = sample.image_path.stem
        source_dir = metadata.get(stem, "")
        if source_dir in oversample_sources:
            weights.append(float(args.oversample_factor))
            boosted_stems.append(stem)
        else:
            weights.append(1.0)

    if not boosted_stems:
        return None, []

    weight_tensor = torch.as_tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=weight_tensor,
        num_samples=len(weights),
        replacement=True,
    )
    return sampler, boosted_stems


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    *,
    train: bool,
    use_amp: bool,
    tversky_alpha: float,
    tversky_beta: float,
    sample_source_lookup: dict[str, str] | None = None,
    focus_sources: set[str] | None = None,
):
    import torch

    model.train(train)
    total_loss = 0.0
    total_dice = 0.0
    total_tversky = 0.0
    total_samples = 0
    total_steps = 0
    focus_dice_sum = 0.0
    focus_tversky_sum = 0.0
    focus_sample_count = 0

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
        batch_dice_scores = compute_sample_dice(logits, masks)
        batch_tversky_scores = compute_sample_tversky(
            logits,
            masks,
            alpha=tversky_alpha,
            beta=tversky_beta,
        )
        total_dice += float(sum(batch_dice_scores))
        total_tversky += float(sum(batch_tversky_scores))
        total_samples += len(batch_dice_scores)

        if sample_source_lookup and focus_sources:
            image_paths = batch["image_path"]
            for image_path, dice_score, tversky_score in zip(
                image_paths,
                batch_dice_scores,
                batch_tversky_scores,
            ):
                stem = Path(image_path).stem
                if sample_source_lookup.get(stem, "") in focus_sources:
                    focus_dice_sum += float(dice_score)
                    focus_tversky_sum += float(tversky_score)
                    focus_sample_count += 1
        total_steps += 1

    if total_steps == 0 or total_samples == 0:
        return {
            "loss": 0.0,
            "dice": 0.0,
            "tversky": 0.0,
            "focus_dice": None,
            "focus_tversky": None,
            "focus_count": 0,
        }
    return {
        "loss": total_loss / total_steps,
        "dice": total_dice / total_samples,
        "tversky": total_tversky / total_samples,
        "focus_dice": (focus_dice_sum / focus_sample_count) if focus_sample_count > 0 else None,
        "focus_tversky": (focus_tversky_sum / focus_sample_count) if focus_sample_count > 0 else None,
        "focus_count": focus_sample_count,
    }


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader

    train_samples = collect_sample_pairs(Path(args.train_images_dir), Path(args.train_masks_dir))
    val_samples = collect_sample_pairs(Path(args.val_images_dir), Path(args.val_masks_dir))
    train_metadata_path = resolve_train_metadata_path(args)
    train_metadata = load_train_metadata(train_metadata_path)
    val_metadata_path = resolve_val_metadata_path(args)
    val_metadata = load_train_metadata(val_metadata_path)
    focus_val_sources = {s.strip() for s in (args.focus_val_sources or []) if s.strip()}
    available_focus_val_sources = {source for source in val_metadata.values() if source in focus_val_sources}

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
    train_sampler, boosted_stems = build_train_sampler(args, train_samples, train_metadata)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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

    criterion = build_loss(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(not args.no_amp and device.type == "cuda"))

    checkpoint_path = Path(args.checkpoint_path)
    last_checkpoint_path = Path(args.last_checkpoint_path)
    ensure_dir(checkpoint_path.parent)
    ensure_dir(last_checkpoint_path.parent)

    best_val_dice = -1.0
    best_val_tversky = -1.0
    best_metric_value = -1.0
    start_epoch = 1

    print(
        f"학습 설정\n"
        f"- loss: BoundaryWeighted BCE + Focal Tversky(alpha={args.tversky_alpha:.2f}, beta={args.tversky_beta:.2f}, gamma={args.focal_gamma:.2f}, boundary_weight={args.boundary_weight:.2f})\n"
        f"- input_size: {args.input_size}\n"
        f"- batch_size: {args.batch_size}\n"
        f"- lr: {args.lr}\n"
        f"- oversample_sources: {', '.join(args.oversample_sources or []) or '(none)'}\n"
        f"- oversample_factor: {args.oversample_factor:.2f}\n"
        f"- focus_val_sources: {', '.join(sorted(focus_val_sources)) or '(none)'}\n"
        f"- train_metadata: {train_metadata_path.resolve() if train_metadata_path else '(none)'}\n"
        f"- val_metadata: {val_metadata_path.resolve() if val_metadata_path else '(none)'}\n"
        f"- boosted_train_samples: {len(boosted_stems)}"
    )

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint가 없습니다: {resume_path}")

        checkpoint = torch.load(str(resume_path), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if not args.reset_optimizer_on_resume:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scaler_state_dict = checkpoint.get("scaler_state_dict")
            if scaler_state_dict is not None and scaler is not None:
                scaler.load_state_dict(scaler_state_dict)

        best_val_dice = float(checkpoint.get("best_val_dice", -1.0))
        best_val_tversky = float(checkpoint.get("best_val_tversky", -1.0))
        best_metric_value = float(checkpoint.get("best_metric_value", -1.0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(
            f"resume 완료\n"
            f"- checkpoint: {resume_path.resolve()}\n"
            f"- 다음 epoch: {start_epoch}\n"
            f"- best_val_dice: {best_val_dice:.4f}\n"
            f"- best_val_tversky: {best_val_tversky:.4f}\n"
            f"- optimizer reset: {'yes' if args.reset_optimizer_on_resume else 'no'}"
        )

    best_metric_name = resolve_best_metric_name(args, focus_sources=available_focus_val_sources)

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            train=True,
            use_amp=(not args.no_amp),
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            scaler,
            train=False,
            use_amp=False,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
            sample_source_lookup=val_metadata,
            focus_sources=focus_val_sources,
        )

        train_loss = train_metrics["loss"]
        train_dice = train_metrics["dice"]
        train_tversky = train_metrics["tversky"]
        val_loss = val_metrics["loss"]
        val_dice = val_metrics["dice"]
        val_tversky = val_metrics["tversky"]
        focus_val_dice = val_metrics["focus_dice"]
        focus_val_tversky = val_metrics["focus_tversky"]

        selection_metric_value = {
            "val_dice": val_dice,
            "val_tversky": val_tversky,
            "focus_val_dice": focus_val_dice if focus_val_dice is not None else -1.0,
            "focus_val_tversky": focus_val_tversky if focus_val_tversky is not None else -1.0,
        }[best_metric_name]

        checkpoint = {
            "epoch": epoch,
            "encoder_name": resolved_encoder,
            "architecture": config_ai.ARCHITECTURE,
            "input_size": args.input_size,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_val_dice": max(best_val_dice, val_dice),
            "best_val_tversky": max(best_val_tversky, val_tversky),
            "train_loss": train_loss,
            "train_dice": train_dice,
            "train_tversky": train_tversky,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_tversky": val_tversky,
            "focus_val_dice": focus_val_dice,
            "focus_val_tversky": focus_val_tversky,
            "focus_val_sources": sorted(focus_val_sources),
            "best_metric_name": best_metric_name,
            "best_metric_value": max(best_metric_value, selection_metric_value),
        }
        torch.save(checkpoint, str(last_checkpoint_path))
        if args.save_every > 0 and epoch % args.save_every == 0:
            periodic_path = build_periodic_checkpoint_path(checkpoint_path, epoch)
            torch.save(checkpoint, str(periodic_path))
        if selection_metric_value > best_metric_value:
            best_metric_value = selection_metric_value
            torch.save(checkpoint, str(checkpoint_path))
        if val_dice > best_val_dice:
            best_val_dice = val_dice
        best_val_tversky = max(best_val_tversky, val_tversky)

        focus_msg = ""
        if focus_val_dice is not None and focus_val_tversky is not None:
            focus_msg = (
                f" focus_val_dice={focus_val_dice:.4f}"
                f" focus_val_tversky={focus_val_tversky:.4f}"
            )
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} train_tversky={train_tversky:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_tversky={val_tversky:.4f} "
            f"best_val_dice={best_val_dice:.4f} best_val_tversky={best_val_tversky:.4f}"
            f"{focus_msg} best_metric={best_metric_name}:{best_metric_value:.4f}"
        )

    print(
        f"\n학습 완료\n"
        f"- best checkpoint: {checkpoint_path.resolve()}\n"
        f"- last checkpoint: {last_checkpoint_path.resolve()}"
    )


if __name__ == "__main__":
    main()
