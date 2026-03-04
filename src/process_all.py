import argparse
from pathlib import Path

import cv2
import numpy as np

INPUT_DIR = "data"
OUTPUT_DIR = "output/coil_only"
MAX_IMAGES = 20
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

MIN_CONTOUR_AREA_RATIO = 0.01
MIN_HOLE_AREA_RATIO = 0.001
BORDER_PENALTY = 0.35
DEFAULT_TRIM_LEVEL = 0
DEFAULT_SMOOTH_LEVEL = 2

ENERGY_KERNEL_RATIO = 0.005
OPEN_KERNEL_RATIO = 0.006
CLOSE_KERNEL_RATIO = 0.012
RECON_KERNEL_RATIO = 0.006
FINAL_OPEN_KERNEL_RATIO = 0.004
TRIM_KERNEL_RATIO = 0.0045
SMOOTH_KERNEL_RATIO = 0.003
HOLE_DILATE_RATIO = 0.0025
INNER_RECT_W_SCALE_FROM_HOLE = 1.00
INNER_RECT_H_SCALE_FROM_HOLE = 0.90
INNER_RECT_FALLBACK_W_RATIO = 0.45
INNER_RECT_FALLBACK_H_RATIO = 0.29
INNER_RECT_CORNER_RATIO = 0.22


def odd_int(v: int, min_value: int = 3) -> int:
    x = max(min_value, int(v))
    if x % 2 == 0:
        x += 1
    return x


def kernel_from_ratio(short_side: int, ratio: float, min_value: int = 3) -> np.ndarray:
    k = odd_int(int(short_side * ratio), min_value=min_value)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def contour_touches_border(cnt: np.ndarray, width: int, height: int, margin: int = 2) -> bool:
    x, y, w, h = cv2.boundingRect(cnt)
    return (
        x <= margin
        or y <= margin
        or x + w >= width - margin
        or y + h >= height - margin
    )


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask)
    out[labels == largest_label] = 255
    return out


def extract_largest_internal_hole(mask: np.ndarray, min_area: int = 0) -> np.ndarray:
    """
    Return a binary mask of the largest hole completely enclosed by foreground.
    """
    mask_bin = np.where(mask > 0, 255, 0).astype(np.uint8)
    h, w = mask_bin.shape[:2]
    inv = cv2.bitwise_not(mask_bin)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask_bin)

    border_labels: set[int] = set()
    border_labels.update(labels[0, :].tolist())
    border_labels.update(labels[h - 1, :].tolist())
    border_labels.update(labels[:, 0].tolist())
    border_labels.update(labels[:, w - 1].tolist())

    best_label = -1
    best_area = 0
    for lab in range(1, num_labels):
        if lab in border_labels:
            continue
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= min_area and area > best_area:
            best_area = area
            best_label = lab

    if best_label == -1:
        return np.zeros_like(mask_bin)

    hole = np.zeros_like(mask_bin)
    hole[labels == best_label] = 255
    return hole


def draw_rounded_rect(
    shape: tuple[int, int],
    center: tuple[float, float],
    size: tuple[int, int],
    corner_ratio: float,
) -> np.ndarray:
    h, w = shape
    cx, cy = center
    rw, rh = size

    rw = int(np.clip(rw, 8, w))
    rh = int(np.clip(rh, 8, h))
    x1 = int(np.clip(round(cx - rw / 2), 0, w - 1))
    y1 = int(np.clip(round(cy - rh / 2), 0, h - 1))
    x2 = int(np.clip(round(cx + rw / 2), 0, w - 1))
    y2 = int(np.clip(round(cy + rh / 2), 0, h - 1))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((h, w), dtype=np.uint8)

    rect_w = x2 - x1 + 1
    rect_h = y2 - y1 + 1
    radius = int(min(rect_w, rect_h) * corner_ratio)
    radius = max(2, radius)
    radius = min(radius, rect_w // 2 - 1, rect_h // 2 - 1)
    if radius < 2:
        radius = 2

    out = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(out, (x1 + radius, y1), (x2 - radius, y2), 255, cv2.FILLED)
    cv2.rectangle(out, (x1, y1 + radius), (x2, y2 - radius), 255, cv2.FILLED)
    cv2.circle(out, (x1 + radius, y1 + radius), radius, 255, cv2.FILLED)
    cv2.circle(out, (x2 - radius, y1 + radius), radius, 255, cv2.FILLED)
    cv2.circle(out, (x1 + radius, y2 - radius), radius, 255, cv2.FILLED)
    cv2.circle(out, (x2 - radius, y2 - radius), radius, 255, cv2.FILLED)
    return out


def build_inner_rounded_rect_hole(
    hole_template: np.ndarray,
    outer_contour: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    h, w = shape

    if np.any(hole_template):
        hole_contours, _ = cv2.findContours(hole_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if hole_contours:
            hole_cnt = max(hole_contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(hole_cnt)
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            rw = int(bw * INNER_RECT_W_SCALE_FROM_HOLE)
            rh = int(bh * INNER_RECT_H_SCALE_FROM_HOLE)
            return draw_rounded_rect((h, w), (cx, cy), (rw, rh), INNER_RECT_CORNER_RATIO)

    x, y, bw, bh = cv2.boundingRect(outer_contour)
    moments = cv2.moments(outer_contour)
    if moments["m00"] > 1e-6:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx = x + bw / 2.0
        cy = y + bh / 2.0

    rw = int(bw * INNER_RECT_FALLBACK_W_RATIO)
    rh = int(bh * INNER_RECT_FALLBACK_H_RATIO)
    return draw_rounded_rect((h, w), (cx, cy), (rw, rh), INNER_RECT_CORNER_RATIO)


def trim_mask(mask: np.ndarray, short_side: int, trim_level: int) -> np.ndarray:
    if trim_level <= 0 or not np.any(mask):
        return mask

    k = odd_int(int(short_side * (TRIM_KERNEL_RATIO + 0.0012 * trim_level)), min_value=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    if trim_level <= 2:
        erode_iter = 1
    elif trim_level <= 5:
        erode_iter = 2
    else:
        erode_iter = 3

    trimmed = cv2.erode(mask, kernel, iterations=erode_iter)

    # 얇은 잔가지 제거를 위한 추가 open
    k2 = odd_int(max(3, k // 2), min_value=3)
    k2_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
    trimmed = cv2.morphologyEx(trimmed, cv2.MORPH_OPEN, k2_kernel)
    trimmed = keep_largest_component(trimmed)

    if not np.any(trimmed):
        return mask

    # 과도한 손실 방지: trim_level이 높아도 최소 15%는 유지
    before_area = int(np.count_nonzero(mask))
    after_area = int(np.count_nonzero(trimmed))
    if after_area < int(before_area * 0.15):
        return cv2.erode(mask, kernel, iterations=1)
    return trimmed


def smooth_mask_boundary(mask: np.ndarray, short_side: int, smooth_level: int) -> np.ndarray:
    if smooth_level <= 0 or not np.any(mask):
        return mask

    blur_k = odd_int(int(short_side * (SMOOTH_KERNEL_RATIO + 0.0008 * smooth_level)), min_value=3)
    blurred = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # 경계 톱니를 줄이되 본체 손실은 최소화
    morph_k = odd_int(max(3, blur_k // 2), min_value=3)
    mk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_k, morph_k))
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, mk)
    smoothed = keep_largest_component(smoothed)

    # 고강도 스무딩에서는 외곽 컨투어 자체를 평활화하여 톱니를 더 줄입니다.
    contours, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return smoothed

    largest = max(contours, key=cv2.contourArea)
    pts = largest[:, 0, :].astype(np.float32)
    n = len(pts)
    if n < 24:
        return smoothed

    win = odd_int(3 + (smooth_level * 4), min_value=3)
    half = win // 2
    iters = 1 + (smooth_level // 2)

    for _ in range(iters):
        pad = np.pad(pts, ((half, half), (0, 0)), mode="wrap")
        k = np.ones(win, dtype=np.float32) / float(win)
        xs = np.convolve(pad[:, 0], k, mode="valid")
        ys = np.convolve(pad[:, 1], k, mode="valid")
        pts = np.stack([xs, ys], axis=1)

    h, w = smoothed.shape[:2]
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    smooth_cnt = np.round(pts).astype(np.int32).reshape(-1, 1, 2)

    contour_smooth = np.zeros_like(smoothed)
    cv2.drawContours(contour_smooth, [smooth_cnt], -1, 255, cv2.FILLED)
    contour_smooth = keep_largest_component(contour_smooth)

    # 지나친 형태 붕괴 방지
    before_area = int(np.count_nonzero(smoothed))
    after_area = int(np.count_nonzero(contour_smooth))
    if before_area > 0 and after_area < int(before_area * 0.55):
        return smoothed
    return contour_smooth


def opening_by_reconstruction(mask: np.ndarray, kernel: np.ndarray, max_iters: int = 64) -> np.ndarray:
    marker = cv2.erode(mask, kernel, iterations=1)
    for _ in range(max_iters):
        dil = cv2.dilate(marker, kernel, iterations=1)
        new_marker = cv2.min(dil, mask)
        if np.array_equal(new_marker, marker):
            break
        marker = new_marker
    return marker


def choose_best_outer_contour(
    contours: list[np.ndarray],
    hierarchy: np.ndarray,
    width: int,
    height: int,
    image_area: int,
) -> int:
    outer_indices = [i for i in range(len(contours)) if hierarchy[i][3] == -1]
    if not outer_indices:
        return -1

    min_area = image_area * MIN_CONTOUR_AREA_RATIO
    best_idx = -1
    best_score = -1.0

    for i in outer_indices:
        area = float(cv2.contourArea(contours[i]))
        if area < min_area:
            continue

        touch = contour_touches_border(contours[i], width, height)
        hull = cv2.convexHull(contours[i])
        hull_area = float(cv2.contourArea(hull))
        solidity = (area / hull_area) if hull_area > 1e-6 else 0.0

        score = area * (BORDER_PENALTY if touch else 1.0) * (0.7 + 0.3 * solidity)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx != -1:
        return best_idx
    return max(outer_indices, key=lambda i: cv2.contourArea(contours[i]))


def apply_texture_mask(
    img: np.ndarray,
    trim_level: int = DEFAULT_TRIM_LEVEL,
    smooth_level: int = DEFAULT_SMOOTH_LEVEL,
) -> np.ndarray | None:
    h, w = img.shape[:2]
    short_side = min(h, w)
    image_area = h * w

    # 1. 흑백 변환 + 엣지 강도
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)

    # 2. 텍스처 에너지 맵 생성 (해상도 비례 커널)
    energy_k = odd_int(int(short_side * ENERGY_KERNEL_RATIO), min_value=3)
    energy = cv2.boxFilter(magnitude, -1, (energy_k, energy_k))
    energy_8u = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 3. 자동 이진화 (Otsu)
    _, mask = cv2.threshold(energy_8u, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 4. 형태 정리 (해상도 비례 Open -> Close)
    kernel_open = kernel_from_ratio(short_side, OPEN_KERNEL_RATIO)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    kernel_close = kernel_from_ratio(short_side, CLOSE_KERNEL_RATIO)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # 5. 윤곽선 + 계층 검출
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        return None

    hierarchy = hierarchy[0]
    outer_idx = choose_best_outer_contour(contours, hierarchy, w, h, image_area)
    if outer_idx == -1:
        return None

    outer_area = float(cv2.contourArea(contours[outer_idx]))
    if outer_area <= 0:
        return None

    # 6. 선택된 코일 외곽만 채우기
    body_mask = np.zeros_like(gray)
    cv2.drawContours(body_mask, contours, outer_idx, 255, cv2.FILLED)

    # 7. 내부 구멍 제거: 가장 큰 1개가 아니라 큰 구멍들을 모두 반영
    child_idx = hierarchy[outer_idx][2]
    min_hole_area = max(image_area * MIN_HOLE_AREA_RATIO, outer_area * 0.005)
    if child_idx != -1:
        while child_idx != -1:
            area = cv2.contourArea(contours[child_idx])
            if area >= min_hole_area:
                cv2.drawContours(body_mask, contours, child_idx, 0, cv2.FILLED)
            child_idx = hierarchy[child_idx][0]

    # 내부 구멍 템플릿 보존: 후처리에서 구멍이 메워지면 다시 파냅니다.
    hole_template = extract_largest_internal_hole(
        body_mask,
        min_area=int(image_area * MIN_HOLE_AREA_RATIO),
    )

    # 8. 돌출/가지 제거: opening-by-reconstruction (본체 보존, 얇은 돌기 제거)
    recon_kernel = kernel_from_ratio(short_side, RECON_KERNEL_RATIO)
    refined = opening_by_reconstruction(body_mask, recon_kernel)

    # 9. 마지막 가벼운 정리 + 최대 컴포넌트 보장
    final_kernel = kernel_from_ratio(short_side, FINAL_OPEN_KERNEL_RATIO)
    final_mask = cv2.morphologyEx(refined, cv2.MORPH_OPEN, final_kernel)
    final_mask = keep_largest_component(final_mask)
    final_mask = trim_mask(final_mask, short_side, trim_level)
    final_mask = smooth_mask_boundary(final_mask, short_side, smooth_level)

    # 내부 컷은 둥근 직사각형으로 고정합니다.
    rounded_hole = build_inner_rounded_rect_hole(
        hole_template=hole_template,
        outer_contour=contours[outer_idx],
        shape=final_mask.shape[:2],
    )
    if np.any(rounded_hole):
        hole_k = kernel_from_ratio(short_side, HOLE_DILATE_RATIO)
        hole_keep = cv2.dilate(rounded_hole, hole_k, iterations=1)
        hole_keep = cv2.bitwise_and(hole_keep, final_mask)
        final_mask[hole_keep > 0] = 0
        final_mask = keep_largest_component(final_mask)

    if not np.any(final_mask):
        return None

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
    trim_level: int = DEFAULT_TRIM_LEVEL,
    smooth_level: int = DEFAULT_SMOOTH_LEVEL,
) -> None:
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"[ERROR] 입력 폴더 없음: {in_dir.resolve()}")
        return

    img_files = collect_image_files(in_dir, MAX_IMAGES)
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

        result = apply_texture_mask(
            img=img,
            trim_level=trim_level,
            smooth_level=smooth_level,
        )
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
        description="Edge/Texture 기반 코일 정밀 마스킹 배치 처리 (입/출력/개수 고정)"
    )
    parser.add_argument(
        "--trim-level",
        type=int,
        default=DEFAULT_TRIM_LEVEL,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        help="추가 깎기 강도 (0=끄기, 기본=0, 8=최강)",
    )
    parser.add_argument(
        "--smooth-level",
        type=int,
        default=DEFAULT_SMOOTH_LEVEL,
        choices=[0, 1, 2, 3, 4, 5],
        help="경계 매끄럽게 보정 강도 (0=끄기, 기본=2)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_mask_batch(
        trim_level=args.trim_level,
        smooth_level=args.smooth_level,
    )
