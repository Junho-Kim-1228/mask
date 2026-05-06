from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config_ai
from io_utils_ai import resize_with_padding, restore_from_padding


@dataclass
class SegmentationResult:
    probability: np.ndarray
    mask: np.ndarray
    score: float
    meta: dict[str, Any] = field(default_factory=dict)


def _import_runtime():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch가 필요합니다. `pip install torch` 후 다시 시도하세요.") from exc

    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError(
            "segmentation_models_pytorch가 필요합니다. `pip install segmentation-models-pytorch timm` 후 다시 시도하세요."
        ) from exc
    return torch, smp


def resolve_device(device: str):
    torch, _ = _import_runtime()
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _encoder_candidates(requested: str) -> list[str]:
    candidates = [requested]
    for candidate in config_ai.ENCODER_NAME_FALLBACKS:
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def build_unetplusplus_model(
    *,
    encoder_name: str = config_ai.ENCODER_NAME,
    encoder_weights: str | None = None,
):
    _, smp = _import_runtime()
    errors: list[str] = []
    for candidate in _encoder_candidates(encoder_name):
        try:
            model = smp.UnetPlusPlus(
                encoder_name=candidate,
                encoder_weights=encoder_weights,
                in_channels=config_ai.IN_CHANNELS,
                classes=config_ai.CLASSES,
                activation=config_ai.ACTIVATION,
            )
            return model, candidate
        except Exception as exc:  # pragma: no cover - backend specific
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError(
        "EfficientNet-B4 encoder로 U-Net++ 모델을 생성하지 못했습니다. "
        + " | ".join(errors)
    )


def _strip_common_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
    prefixes = ("module.", "model.")
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return _strip_common_prefixes(checkpoint[key])
    if isinstance(checkpoint, dict):
        return _strip_common_prefixes(checkpoint)
    raise RuntimeError("체크포인트 형식을 해석할 수 없습니다.")


class CoilSegmenter:
    def __init__(
        self,
        model_path: Path,
        *,
        device: str = config_ai.DEVICE,
        input_size: int = config_ai.INPUT_SIZE,
        encoder_name: str = config_ai.ENCODER_NAME,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"모델 가중치가 없습니다: {self.model_path}\n"
                f"예시 경로: {config_ai.MODEL_PATH}"
            )

        torch, _ = _import_runtime()
        self.torch = torch
        self.device = resolve_device(device)
        self.input_size = int(input_size)
        self.model, self.encoder_name = build_unetplusplus_model(
            encoder_name=encoder_name,
            encoder_weights=None,
        )
        checkpoint = torch.load(str(self.model_path), map_location=self.device)
        state_dict = _extract_state_dict(checkpoint)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.load_meta = {
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }

    def _preprocess(self, image_bgr: np.ndarray):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized, meta = resize_with_padding(image_rgb, self.input_size, is_mask=False, pad_value=0)
        image = resized.astype(np.float32) / 255.0
        mean = np.array(config_ai.IMAGE_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(config_ai.IMAGE_STD, dtype=np.float32).reshape(1, 1, 3)
        image = (image - mean) / std
        chw = np.transpose(image, (2, 0, 1))
        tensor = self.torch.from_numpy(chw).unsqueeze(0).float().to(self.device)
        return tensor, meta

    def predict_probability(self, image_bgr: np.ndarray) -> np.ndarray:
        tensor, meta = self._preprocess(image_bgr)
        with self.torch.inference_mode():
            logits = self.model(tensor)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probability = self.torch.sigmoid(logits)[0, 0].detach().float().cpu().numpy()
        restored = restore_from_padding(probability, meta, is_mask=False)
        return np.clip(restored.astype(np.float32), 0.0, 1.0)

    def predict(
        self,
        image_bgr: np.ndarray,
        *,
        confidence_threshold: float,
        mask_threshold: float,
    ) -> SegmentationResult:
        probability = self.predict_probability(image_bgr)
        score = float(np.percentile(probability, config_ai.PREDICTION_SCORE_PERCENTILE))
        if score < confidence_threshold:
            mask = np.zeros_like(probability, dtype=np.uint8)
        else:
            mask = np.where(probability >= mask_threshold, 255, 0).astype(np.uint8)
        return SegmentationResult(
            probability=probability,
            mask=mask,
            score=score,
            meta={
                "device": str(self.device),
                "encoder_name": self.encoder_name,
                "input_size": self.input_size,
                **self.load_meta,
            },
        )


def build_segmenter(
    *,
    model_path: Path,
    device: str = config_ai.DEVICE,
    input_size: int = config_ai.INPUT_SIZE,
    encoder_name: str = config_ai.ENCODER_NAME,
) -> CoilSegmenter:
    return CoilSegmenter(
        model_path=model_path,
        device=device,
        input_size=input_size,
        encoder_name=encoder_name,
    )
