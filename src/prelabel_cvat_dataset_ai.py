from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2

import config_ai
from io_utils_ai import ensure_dir, list_image_files, load_image_bgr
from postprocess_ai import PostprocessConfig, postprocess_probability_map
from segment_model import build_segmenter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="학습된 AI 모델로 raw 이미지 폴더를 CVAT 재라벨링용 데이터셋으로 변환합니다."
    )
    parser.add_argument("--input-dir", default="dataset")
    parser.add_argument("--output-dir", default="2.0ds")
    parser.add_argument("--model-path", default=str(config_ai.MODEL_PATH))
    parser.add_argument("--device", default=config_ai.DEVICE)
    parser.add_argument("--input-size", type=int, default=config_ai.INPUT_SIZE)
    parser.add_argument("--encoder-name", default=config_ai.ENCODER_NAME)
    parser.add_argument("--confidence-threshold", type=float, default=config_ai.CONF_THRESHOLD)
    parser.add_argument("--mask-threshold", type=float, default=config_ai.MASK_THRESHOLD)
    parser.add_argument("--min-component-area", type=int, default=config_ai.MIN_COMPONENT_AREA)
    parser.add_argument("--morph-open-kernel", type=int, default=config_ai.MORPH_OPEN_KERNEL)
    parser.add_argument("--morph-close-kernel", type=int, default=config_ai.MORPH_CLOSE_KERNEL)
    parser.add_argument("--min-hole-area", type=int, default=config_ai.MIN_HOLE_AREA)
    parser.add_argument("--no-keep-largest-component", action="store_true")
    parser.add_argument("--no-preserve-inner-holes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_cvat_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "images": output_dir / "JPEGImages",
        "seg_class": output_dir / "SegmentationClass",
        "seg_object": output_dir / "SegmentationObject",
        "imageset": output_dir / "ImageSets" / "Segmentation",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_labelmap(output_dir: Path) -> None:
    labelmap = output_dir / "labelmap.txt"
    labelmap.write_text(
        "# label : color (RGB) : 'body' parts : actions\n"
        "background:0,0,0::\n"
        "coil:255,255,255::\n",
        encoding="utf-8",
    )


def build_post_cfg(args: argparse.Namespace) -> PostprocessConfig:
    return PostprocessConfig(
        confidence_threshold=args.confidence_threshold,
        mask_threshold=args.mask_threshold,
        min_component_area=args.min_component_area,
        morph_open_kernel=args.morph_open_kernel,
        morph_close_kernel=args.morph_close_kernel,
        keep_largest_component=not args.no_keep_largest_component,
        preserve_inner_holes=not args.no_preserve_inner_holes,
        min_hole_area=args.min_hole_area,
    )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    paths = ensure_cvat_dirs(output_dir)
    write_labelmap(output_dir)

    image_paths = list_image_files(input_dir)
    segmenter = build_segmenter(
        model_path=Path(args.model_path),
        device=args.device,
        input_size=args.input_size,
        encoder_name=args.encoder_name,
    )
    post_cfg = build_post_cfg(args)

    saved_stems: list[str] = []
    ok = 0
    fail = 0
    skip = 0
    for idx, image_path in enumerate(image_paths, start=1):
        try:
            image = load_image_bgr(image_path)
            probability = segmenter.predict_probability(image)
            final_mask = postprocess_probability_map(probability, post_cfg)
            if not final_mask.any():
                print(f"[SKIP {idx}/{len(image_paths)}] 마스크 없음: {image_path.name}")
                skip += 1
                continue

            dst_image = paths["images"] / image_path.name
            dst_seg_class = paths["seg_class"] / f"{image_path.stem}.png"
            dst_seg_object = paths["seg_object"] / f"{image_path.stem}.png"
            if args.overwrite or not dst_image.exists():
                shutil.copy2(image_path, dst_image)

            final_mask_index = (final_mask > 0).astype("uint8")
            if not cv2.imwrite(str(dst_seg_class), final_mask_index):
                raise RuntimeError(f"SegmentationClass 저장 실패: {dst_seg_class}")
            if not cv2.imwrite(str(dst_seg_object), final_mask_index):
                raise RuntimeError(f"SegmentationObject 저장 실패: {dst_seg_object}")

            saved_stems.append(image_path.stem)
            print(f"[OK {idx}/{len(image_paths)}] {image_path.name}")
            ok += 1
        except Exception as exc:
            print(f"[FAIL {idx}/{len(image_paths)}] {image_path.name} | {exc}")
            fail += 1

    default_txt = paths["imageset"] / "default.txt"
    default_txt.write_text(
        "\n".join(saved_stems) + ("\n" if saved_stems else ""),
        encoding="utf-8",
    )

    print(
        f"\n프리라벨 완료\n"
        f"- input: {input_dir.resolve()}\n"
        f"- output: {output_dir.resolve()}\n"
        f"- ok: {ok}\n"
        f"- skip: {skip}\n"
        f"- fail: {fail}"
    )


if __name__ == "__main__":
    main()
