from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import config_ai


@dataclass(frozen=True)
class ResizeMeta:
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    top: int
    bottom: int
    left: int
    right: int
    target_size: int


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_image_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"입력 폴더가 없습니다: {input_dir}")
    return [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in config_ai.VALID_EXTENSIONS
    ]


def load_image_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"이미지 로드 실패: {path}")
    return image


def load_mask_grayscale(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"마스크 로드 실패: {path}")
    return mask


def find_mask_path(mask_dir: Path, image_stem: str) -> Path | None:
    for ext in config_ai.VALID_EXTENSIONS:
        candidate = mask_dir / f"{image_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def resize_with_padding(
    image: np.ndarray,
    target_size: int,
    *,
    is_mask: bool = False,
    pad_value: int | tuple[int, int, int] = 0,
) -> tuple[np.ndarray, ResizeMeta]:
    if image.ndim not in {2, 3}:
        raise ValueError("2D 또는 3D 배열만 지원합니다.")

    height, width = image.shape[:2]
    scale = min(target_size / max(height, 1), target_size / max(width, 1))
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)

    top = (target_size - resized_height) // 2
    bottom = target_size - resized_height - top
    left = (target_size - resized_width) // 2
    right = target_size - resized_width - left

    bordered = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value,
    )
    meta = ResizeMeta(
        original_height=height,
        original_width=width,
        resized_height=resized_height,
        resized_width=resized_width,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        target_size=target_size,
    )
    return bordered, meta


def restore_from_padding(image: np.ndarray, meta: ResizeMeta, *, is_mask: bool = False) -> np.ndarray:
    cropped = image[meta.top:meta.top + meta.resized_height, meta.left:meta.left + meta.resized_width]
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    restored = cv2.resize(
        cropped,
        (meta.original_width, meta.original_height),
        interpolation=interpolation,
    )
    return restored


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    return cv2.bitwise_and(image, image, mask=mask_u8)


def save_binary_mask(mask_path: Path, mask: np.ndarray) -> None:
    mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not cv2.imwrite(str(mask_path), mask_u8):
        raise RuntimeError(f"마스크 저장 실패: {mask_path}")


def colorize_probability_map(probability_map: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability_map, 0.0, 1.0)
    heatmap = (clipped * 255.0).astype(np.uint8)
    return cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)


def draw_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float = 0.60,
    contour_color: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    overlay = image.copy()
    mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not np.any(mask_u8):
        return overlay

    dark = (overlay.astype(np.float32) * 0.35).astype(np.uint8)
    overlay[:] = dark
    overlay[mask_u8 > 0] = cv2.addWeighted(
        image[mask_u8 > 0],
        1.0 - alpha,
        image[mask_u8 > 0],
        alpha,
        0,
    )
    overlay[mask_u8 > 0] = image[mask_u8 > 0]

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, contour_color, 2, cv2.LINE_AA)
    return overlay


def fit_contain(image: np.ndarray, size: tuple[int, int], pad_value: int = 20) -> np.ndarray:
    target_width, target_height = size
    src_height, src_width = image.shape[:2]
    scale = min(target_width / max(src_width, 1), target_height / max(src_height, 1))
    new_width = max(1, int(round(src_width * scale)))
    new_height = max(1, int(round(src_height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_height, target_width, 3), pad_value, dtype=np.uint8)
    x0 = max(0, (target_width - new_width) // 2)
    y0 = max(0, (target_height - new_height) // 2)
    canvas[y0:y0 + new_height, x0:x0 + new_width] = resized
    cv2.rectangle(canvas, (x0, y0), (x0 + new_width - 1, y0 + new_height - 1), (120, 120, 120), 1)
    return canvas
