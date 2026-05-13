from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import config_ai
from io_utils_ai import (
    apply_mask_to_image,
    crop_to_bbox,
    ensure_dir,
    expand_bbox,
    list_image_files,
    load_image_bgr,
    mask_to_bbox,
    paste_mask_from_bbox,
)
from postprocess_ai import PostprocessConfig, postprocess_probability_map
from segment_model import build_segmenter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="학습된 AI 모델로 원본 이미지에서 코일만 남긴 전처리 결과를 저장합니다."
    )
    parser.add_argument("--input-dir", default="dataset")
    parser.add_argument("--output-dir", default=str(config_ai.OUTPUT_DIR))
    parser.add_argument("--model-path", default=str(config_ai.MODEL_PATH))
    parser.add_argument("--device", default=config_ai.DEVICE)
    parser.add_argument("--input-size", type=int, default=config_ai.INPUT_SIZE)
    parser.add_argument("--encoder-name", default=config_ai.ENCODER_NAME)
    parser.add_argument("--confidence-threshold", type=float, default=config_ai.CONF_THRESHOLD)
    parser.add_argument("--mask-threshold", type=float, default=config_ai.MASK_THRESHOLD)
    parser.add_argument("--min-component-area", type=int, default=config_ai.MIN_COMPONENT_AREA)
    parser.add_argument("--morph-open-kernel", type=int, default=config_ai.MORPH_OPEN_KERNEL)
    parser.add_argument("--morph-close-kernel", type=int, default=config_ai.MORPH_CLOSE_KERNEL)
    parser.add_argument("--outer-recover-kernel", type=int, default=config_ai.OUTER_RECOVER_KERNEL)
    parser.add_argument("--refine-crop-margin-ratio", type=float, default=0.0)
    parser.add_argument("--refine-crop-min-margin", type=int, default=8)
    parser.add_argument("--min-hole-area", type=int, default=config_ai.MIN_HOLE_AREA)
    parser.add_argument("--no-keep-largest-component", action="store_true")
    parser.add_argument("--no-preserve-inner-holes", action="store_true")
    parser.add_argument("--save-mask", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_post_cfg(args: argparse.Namespace) -> PostprocessConfig:
    return PostprocessConfig(
        confidence_threshold=args.confidence_threshold,
        mask_threshold=args.mask_threshold,
        min_component_area=args.min_component_area,
        morph_open_kernel=args.morph_open_kernel,
        morph_close_kernel=args.morph_close_kernel,
        outer_recover_kernel=args.outer_recover_kernel,
        keep_largest_component=not args.no_keep_largest_component,
        preserve_inner_holes=not args.no_preserve_inner_holes,
        min_hole_area=args.min_hole_area,
    )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(Path(args.output_dir))
    mask_dir = ensure_dir(output_dir / "masks") if args.save_mask else None

    image_paths = list_image_files(input_dir)
    segmenter = build_segmenter(
        model_path=Path(args.model_path),
        device=args.device,
        input_size=args.input_size,
        encoder_name=args.encoder_name,
    )
    post_cfg = build_post_cfg(args)

    ok = 0
    skip = 0
    fail = 0

    for idx, image_path in enumerate(image_paths, start=1):
        try:
            masked_path = output_dir / f"{image_path.stem}_masked.bmp"
            mask_path = mask_dir / f"{image_path.stem}.png" if mask_dir is not None else None

            if not args.overwrite and masked_path.exists():
                print(f"[SKIP {idx}/{len(image_paths)}] 이미 존재: {masked_path.name}")
                skip += 1
                continue

            image = load_image_bgr(image_path)
            probability = segmenter.predict_probability(image)
            final_mask = postprocess_probability_map(probability, post_cfg)
            if final_mask.any() and args.refine_crop_margin_ratio > 0:
                bbox = mask_to_bbox(final_mask)
                if bbox is not None:
                    refined_bbox = expand_bbox(
                        bbox,
                        image.shape,
                        margin_ratio=args.refine_crop_margin_ratio,
                        min_margin=args.refine_crop_min_margin,
                    )
                    crop_image = crop_to_bbox(image, refined_bbox)
                    refined_probability = segmenter.predict_probability(crop_image)
                    refined_mask = postprocess_probability_map(refined_probability, post_cfg)
                    if refined_mask.any():
                        refined_mask_full = paste_mask_from_bbox(
                            image.shape,
                            refined_mask,
                            refined_bbox,
                        )
                        final_mask = np.where(
                            (final_mask > 0) | (refined_mask_full > 0),
                            255,
                            0,
                        ).astype("uint8")
            if not final_mask.any():
                print(f"[SKIP {idx}/{len(image_paths)}] 마스크 없음: {image_path.name}")
                skip += 1
                continue

            masked_image = apply_mask_to_image(image, final_mask)
            if not cv2.imwrite(str(masked_path), masked_image):
                raise RuntimeError(f"결과 저장 실패: {masked_path}")

            if mask_path is not None:
                final_mask_u8 = (final_mask > 0).astype("uint8") * 255
                if not cv2.imwrite(str(mask_path), final_mask_u8):
                    raise RuntimeError(f"마스크 저장 실패: {mask_path}")

            print(f"[OK {idx}/{len(image_paths)}] {image_path.name}")
            ok += 1
        except Exception as exc:
            print(f"[FAIL {idx}/{len(image_paths)}] {image_path.name} | {exc}")
            fail += 1

    print(
        f"\n전처리 완료\n"
        f"- input: {input_dir.resolve()}\n"
        f"- output: {output_dir.resolve()}\n"
        f"- ok: {ok}\n"
        f"- skip: {skip}\n"
        f"- fail: {fail}"
    )
    if mask_dir is not None:
        print(f"- mask_dir: {mask_dir.resolve()}")


if __name__ == "__main__":
    main()
