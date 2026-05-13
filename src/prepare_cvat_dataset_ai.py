from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import config_ai


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CVAT VOC-style 데이터셋(예: 1.0ds, 2.1ds)을 train/val 학습 구조로 변환합니다."
    )
    parser.add_argument("--source-dir", action="append", dest="source_dirs")
    parser.add_argument("--output-dir", default="prepared_trainset")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--hard-val-source", action="append", dest="hard_val_sources")
    parser.add_argument("--hard-val-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_stems(source_dir: Path) -> list[str]:
    split_file = source_dir / "ImageSets" / "Segmentation" / "default.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"default.txt가 없습니다: {split_file}")
    stems = [
        line.strip()
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not stems:
        raise RuntimeError(f"default.txt가 비어 있습니다: {split_file}")
    return stems


def find_image_path(images_dir: Path, stem: str) -> Path:
    for ext in sorted(config_ai.VALID_EXTENSIONS):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"원본 이미지를 찾을 수 없습니다: {stem}")


def ensure_split_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "train_images": output_dir / "train" / "images",
        "train_masks": output_dir / "train" / "masks",
        "val_images": output_dir / "val" / "images",
        "val_masks": output_dir / "val" / "masks",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def collect_pairs(source_dirs: list[Path]) -> dict[str, tuple[Path, Path, str]]:
    pairs: dict[str, tuple[Path, Path, str]] = {}
    for source_dir in source_dirs:
        images_dir = source_dir / "JPEGImages"
        masks_dir = source_dir / "SegmentationClass"
        source_name = source_dir.name
        stems = load_stems(source_dir)
        for stem in stems:
            image_path = find_image_path(images_dir, stem)
            mask_path = masks_dir / f"{stem}.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"마스크를 찾을 수 없습니다: {mask_path}")
            pairs[stem] = (image_path, mask_path, source_name)
    return pairs


def copy_pairs(
    stems: list[str],
    pairs: dict[str, tuple[Path, Path, str]],
    out_images: Path,
    out_masks: Path,
    *,
    overwrite: bool,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for stem in stems:
        image_path, mask_path, source_name = pairs[stem]
        dst_image = out_images / image_path.name
        dst_mask = out_masks / mask_path.name
        if overwrite or not dst_image.exists():
            shutil.copy2(image_path, dst_image)
        if overwrite or not dst_mask.exists():
            shutil.copy2(mask_path, dst_mask)
        records.append(
            {
                "stem": stem,
                "image_name": dst_image.name,
                "mask_name": dst_mask.name,
                "source_dir": source_name,
            }
        )
    return records


def write_metadata(records: list[dict[str, str]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["stem", "image_name", "mask_name", "source_dir"])
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    source_dirs = [Path(p) for p in (args.source_dirs or ["1.0ds"])]
    output_dir = Path(args.output_dir)

    pairs = collect_pairs(source_dirs)
    stems = sorted(pairs)
    rnd = random.Random(args.seed)
    shuffled = stems[:]
    rnd.shuffle(shuffled)

    val_count = max(1, int(round(len(shuffled) * args.val_ratio))) if len(shuffled) > 1 else 0
    hard_val_sources = {s.strip() for s in (args.hard_val_sources or []) if s.strip()}
    hard_val_stems: list[str] = []
    if hard_val_sources and args.hard_val_count > 0:
        for source_name in sorted(hard_val_sources):
            source_candidates = [stem for stem in shuffled if pairs[stem][2] == source_name]
            rnd.shuffle(source_candidates)
            take_count = min(args.hard_val_count, len(source_candidates))
            hard_val_stems.extend(source_candidates[:take_count])

    hard_val_stems = sorted(set(hard_val_stems))
    required_val_count = max(val_count, len(hard_val_stems))
    remaining_for_val = [stem for stem in shuffled if stem not in hard_val_stems]
    additional_val_count = max(0, required_val_count - len(hard_val_stems))
    val_stems = sorted(hard_val_stems + remaining_for_val[:additional_val_count])
    train_stems = sorted([stem for stem in shuffled if stem not in set(val_stems)])
    if not train_stems:
        train_stems, val_stems = val_stems, []

    paths = ensure_split_dirs(output_dir)
    train_records = copy_pairs(
        train_stems,
        pairs,
        paths["train_images"],
        paths["train_masks"],
        overwrite=args.overwrite,
    )
    val_records = copy_pairs(
        val_stems,
        pairs,
        paths["val_images"],
        paths["val_masks"],
        overwrite=args.overwrite,
    )
    write_metadata(train_records, output_dir / "train_metadata.csv")
    write_metadata(val_records, output_dir / "val_metadata.csv")

    print(
        f"준비 완료\n"
        f"- source_dirs: {', '.join(str(path.resolve()) for path in source_dirs)}\n"
        f"- unique samples: {len(stems)}\n"
        f"- output: {output_dir.resolve()}\n"
        f"- train: {len(train_stems)}\n"
        f"- val: {len(val_stems)}\n"
        f"- hard_val_sources: {', '.join(sorted(hard_val_sources)) or '(none)'}\n"
        f"- hard_val_count_per_source: {args.hard_val_count}\n"
        f"- train metadata: {(output_dir / 'train_metadata.csv').resolve()}\n"
        f"- val metadata: {(output_dir / 'val_metadata.csv').resolve()}"
    )


if __name__ == "__main__":
    main()
