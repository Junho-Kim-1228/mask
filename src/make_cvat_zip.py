from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="dataset 폴더를 CVAT import용 zip으로 묶습니다."
    )
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-zip", default="dataset_for_cvat.zip")
    return parser.parse_args()


def iter_dataset_files(dataset_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == ".gitkeep":
            continue
        files.append(path)
    return files


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_zip = Path(args.output_zip)

    if not dataset_dir.exists():
        raise SystemExit(f"dataset 폴더가 없습니다: {dataset_dir}")

    required_paths = [
        dataset_dir / "labelmap.txt",
        dataset_dir / "ImageSets" / "Segmentation" / "default.txt",
        dataset_dir / "SegmentationClass",
        dataset_dir / "SegmentationObject",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit("필수 경로가 없습니다:\n- " + "\n- ".join(missing))

    files = iter_dataset_files(dataset_dir)
    if not files:
        raise SystemExit(f"압축할 파일이 없습니다: {dataset_dir}")

    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as zf:
        for path in files:
            arcname = path.relative_to(dataset_dir).as_posix()
            zf.write(path, arcname)

    print(f"[OK] {output_zip.resolve()}")
    print("포함된 루트 항목:")
    print("- labelmap.txt")
    print("- ImageSets/Segmentation/default.txt")
    print("- SegmentationClass/*.png")
    print("- SegmentationObject/*.png")


if __name__ == "__main__":
    main()
