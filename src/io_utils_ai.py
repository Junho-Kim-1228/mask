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


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return x1, y1, x2, y2


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
    *,
    margin_ratio: float = 0.08,
    min_margin: int = 8,
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    mx = max(min_margin, int(round(box_w * margin_ratio)))
    my = max(min_margin, int(round(box_h * margin_ratio)))
    nx1 = max(0, x1 - mx)
    ny1 = max(0, y1 - my)
    nx2 = min(width, x2 + mx)
    ny2 = min(height, y2 + my)
    return nx1, ny1, nx2, ny2


def crop_to_bbox(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2].copy()


def paste_mask_from_bbox(
    image_shape: tuple[int, ...],
    crop_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    full_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    full_mask[y1:y2, x1:x2] = np.where(crop_mask > 0, 255, 0).astype(np.uint8)
    return full_mask


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
