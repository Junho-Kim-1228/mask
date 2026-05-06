from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config_ai
from io_utils_ai import find_mask_path, resize_with_padding


@dataclass(frozen=True)
class SamplePair:
    image_path: Path
    mask_path: Path


def collect_sample_pairs(images_dir: Path, masks_dir: Path) -> list[SamplePair]:
    if not images_dir.exists():
        raise FileNotFoundError(f"이미지 폴더가 없습니다: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"마스크 폴더가 없습니다: {masks_dir}")

    samples: list[SamplePair] = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in config_ai.VALID_EXTENSIONS:
            continue
        mask_path = find_mask_path(masks_dir, image_path.stem)
        if mask_path is None:
            raise FileNotFoundError(f"매칭되는 마스크가 없습니다: {image_path.name}")
        samples.append(SamplePair(image_path=image_path, mask_path=mask_path))

    if not samples:
        raise RuntimeError(f"학습 샘플이 없습니다: {images_dir}")
    return samples


def _manual_to_tensor(image_rgb: np.ndarray, mask: np.ndarray):
    import torch

    image = image_rgb.astype(np.float32) / 255.0
    mean = np.array(config_ai.IMAGE_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(config_ai.IMAGE_STD, dtype=np.float32).reshape(1, 1, 3)
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    mask = mask.astype(np.float32)[None, :, :]
    return torch.from_numpy(image).float(), torch.from_numpy(mask).float()


def build_train_transform(input_size: int):
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        return None

    return A.Compose(
        [
            A.LongestMaxSize(max_size=input_size),
            A.PadIfNeeded(
                min_height=input_size,
                min_width=input_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.97, 1.03),
                translate_percent=(-0.02, 0.02),
                rotate=(-5, 5),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=0.4,
            ),
            A.Normalize(mean=config_ai.IMAGE_MEAN, std=config_ai.IMAGE_STD),
            ToTensorV2(transpose_mask=True),
        ]
    )


def build_eval_transform(input_size: int):
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        return None

    return A.Compose(
        [
            A.LongestMaxSize(max_size=input_size),
            A.PadIfNeeded(
                min_height=input_size,
                min_width=input_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
            ),
            A.Normalize(mean=config_ai.IMAGE_MEAN, std=config_ai.IMAGE_STD),
            ToTensorV2(transpose_mask=True),
        ]
    )


class CoilSegDataset:
    def __init__(self, samples: list[SamplePair], *, transform: Any | None = None, input_size: int = config_ai.INPUT_SIZE) -> None:
        self.samples = samples
        self.transform = transform
        self.input_size = int(input_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        mask_gray = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None:
            raise RuntimeError(f"이미지 로드 실패: {sample.image_path}")
        if mask_gray is None:
            raise RuntimeError(f"마스크 로드 실패: {sample.mask_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = np.where(mask_gray > 0, 1.0, 0.0).astype(np.float32)

        if self.transform is not None:
            transformed = self.transform(image=image_rgb, mask=mask)
            image_tensor = transformed["image"].float()
            mask_tensor = transformed["mask"].float().unsqueeze(0)
        else:
            image_rgb, _ = resize_with_padding(image_rgb, self.input_size, is_mask=False, pad_value=0)
            mask, _ = resize_with_padding(mask, self.input_size, is_mask=True, pad_value=0)
            image_tensor, mask_tensor = _manual_to_tensor(image_rgb, mask)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
        }
