import argparse
from pathlib import Path

import cv2
import numpy as np

INPUT_DIR = "data"
OUTPUT_DIR = "output/coil_only"
MAX_IMAGES = 20
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def apply_texture_mask(img: np.ndarray) -> np.ndarray | None:
    # 1. 흑백 변환 및 엣지 강도 계산
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)

    # 2. 텍스처 에너지 맵 생성
    energy = cv2.boxFilter(magnitude, -1, (11, 11))
    energy_8u = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 3. 자동 이진화 (Otsu)
    _, mask = cv2.threshold(energy_8u, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 4. 초기 절단 (Open) - 커널을 조금 더 키워서 질긴 납땜 부위를 1차로 끊어냅니다.
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # 5. 내부 틈새 메우기 (Close)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # 6. 윤곽선 검출
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return None

    # 7. 가장 큰 윤곽선(코일 바디) 찾기
    largest_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    if cv2.contourArea(contours[largest_idx]) <= 0:
        return None

    body_mask = np.zeros_like(gray)
    cv2.drawContours(body_mask, contours, largest_idx, 255, cv2.FILLED)

    # 8. 내부 구멍 파내기
    child_idx = hierarchy[0][largest_idx][2]
    if child_idx != -1:
        largest_child_idx = -1
        max_child_area = 0.0
        while child_idx != -1:
            area = cv2.contourArea(contours[child_idx])
            if area > max_child_area:
                max_child_area = area
                largest_child_idx = child_idx
            child_idx = hierarchy[0][child_idx][0]

        if largest_child_idx != -1:
            cv2.drawContours(body_mask, contours, largest_child_idx, 0, cv2.FILLED)

    # 9. 핵심 수정: 혹 떼어내기 (Opening on the final mask)
    # 깎아내기(Erode) 대신 열림(Open)을 사용하여, 코일 두께는 갉아먹지 않으면서
    # 테두리 안팎으로 삐죽 튀어나온 찌꺼기 픽셀들만 매끄럽게 잘라냅니다.
    fin_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    final_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, fin_kernel)

    # 10. 원본에 마스크 적용
    return cv2.bitwise_and(img, img, mask=final_mask)

def collect_image_files(in_dir: Path, max_images: int) -> list[Path]:
    img_files = [
        p for p in sorted(in_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    if max_images > 0:
        img_files = img_files[:max_images]
    return img_files


def run_mask_batch(
    input_dir: str = INPUT_DIR,
    output_dir: str = OUTPUT_DIR,
    max_images: int = MAX_IMAGES,
) -> None:
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"[ERROR] 입력 폴더 없음: {in_dir.resolve()}")
        return

    img_files = collect_image_files(in_dir, max_images)
    if not img_files:
        print(f"[ERROR] 처리할 이미지가 없습니다: {in_dir.resolve()}")
        return

    ok, fail, skip = 0, 0, 0
    for idx, img_path in enumerate(img_files, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[FAIL {idx}/{len(img_files)}] 이미지 로드 실패: {img_path.name}")
            fail += 1
            continue

        result = apply_texture_mask(img=img)
        if result is None:
            print(f"[SKIP {idx}/{len(img_files)}] 코일 감지 실패: {img_path.name}")
            skip += 1
            continue

        out_path = out_dir / f"{img_path.stem}_masked.bmp"
        if cv2.imwrite(str(out_path), result):
            print(f"[OK {idx}/{len(img_files)}] {img_path.name} -> {out_path.name}")
            ok += 1
        else:
            print(f"[FAIL {idx}/{len(img_files)}] 저장 실패: {img_path.name}")
            fail += 1

    print(
        f"\n완료: 성공 {ok}, 실패 {fail}, 스킵 {skip}, 대상 {len(img_files)}\n"
        f"- 입력: {in_dir.resolve()}\n"
        f"- 출력: {out_dir.resolve()}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edge/Texture 기반 코일 정밀 마스킹 배치 처리"
    )
    parser.add_argument("--input-dir", default=INPUT_DIR, help="입력 이미지 폴더 경로")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="출력 폴더 경로")
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES, help="최대 처리 수")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_mask_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_images=args.max_images,
    )