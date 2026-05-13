from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import config_ai


@dataclass(slots=True)
class PostprocessConfig:
    confidence_threshold: float = config_ai.CONF_THRESHOLD
    mask_threshold: float = config_ai.MASK_THRESHOLD
    min_component_area: int = config_ai.MIN_COMPONENT_AREA
    morph_open_kernel: int = config_ai.MORPH_OPEN_KERNEL
    morph_close_kernel: int = config_ai.MORPH_CLOSE_KERNEL
    outer_recover_kernel: int = config_ai.OUTER_RECOVER_KERNEL
    keep_largest_component: bool = config_ai.KEEP_LARGEST_COMPONENT
    preserve_inner_holes: bool = config_ai.PRESERVE_INNER_HOLES
    min_hole_area: int = config_ai.MIN_HOLE_AREA


def compute_prediction_score(probability_map: np.ndarray) -> float:
    if probability_map.size == 0:
        return 0.0
    return float(np.percentile(probability_map, config_ai.PREDICTION_SCORE_PERCENTILE))


def normalize_kernel_size(value: int) -> int:
    if value <= 0:
        return 0
    if value % 2 == 0:
        value += 1
    return max(3, value)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask)
    out[labels == largest_label] = 255
    return out


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    out = np.zeros_like(mask)
    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == lab] = 255
    return out


def extract_enclosed_holes(mask: np.ndarray, min_hole_area: int) -> np.ndarray:
    mask_u8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    inv = cv2.bitwise_not(mask_u8)
    height, width = mask_u8.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask_u8)

    border_labels: set[int] = set()
    border_labels.update(labels[0, :].tolist())
    border_labels.update(labels[height - 1, :].tolist())
    border_labels.update(labels[:, 0].tolist())
    border_labels.update(labels[:, width - 1].tolist())

    holes = np.zeros_like(mask_u8)
    for lab in range(1, num_labels):
        if lab in border_labels:
            continue
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= min_hole_area:
            holes[labels == lab] = 255
    return holes


def threshold_probability_map(probability_map: np.ndarray, mask_threshold: float) -> np.ndarray:
    clipped = np.clip(probability_map, 0.0, 1.0)
    return np.where(clipped >= mask_threshold, 255, 0).astype(np.uint8)


def apply_conservative_morphology(mask: np.ndarray, open_kernel: int, close_kernel: int) -> np.ndarray:
    out = mask.copy()
    open_kernel = normalize_kernel_size(open_kernel)
    close_kernel = normalize_kernel_size(close_kernel)

    if open_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    if close_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return out


def apply_outer_recover(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = normalize_kernel_size(kernel_size)
    if kernel_size <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


def postprocess_probability_map(probability_map: np.ndarray, config: PostprocessConfig) -> np.ndarray:
    score = compute_prediction_score(probability_map)
    if score < config.confidence_threshold:
        return np.zeros_like(probability_map, dtype=np.uint8)

    raw_mask = threshold_probability_map(probability_map, config.mask_threshold)
    preserved_holes = (
        extract_enclosed_holes(raw_mask, config.min_hole_area)
        if config.preserve_inner_holes
        else np.zeros_like(raw_mask)
    )

    out = remove_small_components(raw_mask, config.min_component_area)
    out = apply_conservative_morphology(out, config.morph_open_kernel, config.morph_close_kernel)
    out = remove_small_components(out, config.min_component_area)
    if config.keep_largest_component and np.any(out):
        out = keep_largest_component(out)
    if config.outer_recover_kernel > 0 and np.any(out):
        out = apply_outer_recover(out, config.outer_recover_kernel)
        if config.keep_largest_component:
            out = keep_largest_component(out)
    if config.preserve_inner_holes and np.any(preserved_holes):
        out[preserved_holes > 0] = 0
    return out
