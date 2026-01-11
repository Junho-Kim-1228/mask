import cv2
import numpy as np
import os
import re
from pathlib import Path

# -----------------------------------------------------------------------------
# 설정
# -----------------------------------------------------------------------------
INPUT_DIR = "data"
MASK_DIR = "masks"
OUTPUT_DIR = "output"

IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}  # 원본은 뭐든 허용
MASK_EXT = ".bmp"  # 마스크는 bmp 고정

# '-뒤 숫자' -> 마스크 파일 매핑
DIGIT_TO_MASK = {
    "1": "mask_type_A.bmp",
    "2": "mask_type_B.bmp",
    "3": "mask_type_C.bmp",
}

# -----------------------------------------------------------------------------
# 파일명에서 4-1 / 4-2 / 4-3 의 "뒤 숫자" 추출
# -----------------------------------------------------------------------------
def extract_digit_after_dash(filename: str) -> str | None:
    m = re.search(r'_(\d+)-(\d+)\b', filename)
    if not m:
        return None
    return m.group(2)

# -----------------------------------------------------------------------------
# 마스크 로드 + 이진화 + (필요시) 리사이즈
# -----------------------------------------------------------------------------
def load_mask(mask_path: Path, target_hw: tuple[int, int]) -> np.ndarray | None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    th, tw = target_hw
    if mask.shape != (th, tw):
        mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)

    # 혹시 0/255가 아니라도 강제 이진화
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask

# -----------------------------------------------------------------------------
# 배치 처리
# -----------------------------------------------------------------------------
def apply_masks_batch():
    in_dir = Path(INPUT_DIR)
    mask_dir = Path(MASK_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"[ERROR] data 폴더 없음: {in_dir.resolve()}")
        return
    if not mask_dir.exists():
        print(f"[ERROR] masks 폴더 없음: {mask_dir.resolve()}")
        return

    img_files = [p for p in sorted(in_dir.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    if not img_files:
        print(f"[ERROR] 처리할 이미지 없음: {in_dir.resolve()}")
        return

    ok, fail = 0, 0

    for img_path in img_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[FAIL] 이미지 로드 실패: {img_path.name}")
            fail += 1
            continue

        digit = extract_digit_after_dash(img_path.stem)  # 확장자 제외한 stem 기준
        if digit is None:
            print(f"[FAIL] 타입 추출 실패(예: _4-1 패턴 없음): {img_path.name}")
            fail += 1
            continue

        mask_filename = DIGIT_TO_MASK.get(digit)
        if mask_filename is None:
            print(f"[FAIL] 마스크 매핑 없음: {img_path.name} (digit={digit})")
            fail += 1
            continue

        mask_path = mask_dir / mask_filename
        if not mask_path.exists():
            print(f"[FAIL] 마스크 파일 없음: {mask_path.name}")
            fail += 1
            continue

        mask = load_mask(mask_path, (img.shape[0], img.shape[1]))
        if mask is None:
            print(f"[FAIL] 마스크 로드 실패: {mask_path.name}")
            fail += 1
            continue

        # ✅ 마스크 영역만 원본 유지, 나머지 검정
        result = img.copy()
        result[mask == 0] = 0

        # ✅ output은 bmp로 저장
        out_path = out_dir / f"{img_path.stem}_masked.bmp"
        cv2.imwrite(str(out_path), result)

        print(f"[OK] {img_path.name} (4-{digit}) + {mask_path.name} -> {out_path.name}")
        ok += 1

    print(f"\n완료: 성공 {ok}, 실패 {fail}, 입력 {len(img_files)}")

if __name__ == "__main__":
    apply_masks_batch()
