import cv2
import numpy as np
import os

# -----------------------------------------------------------------------------
# 1. 헬퍼 함수들
# -----------------------------------------------------------------------------
def keep_largest_component(mask_bin: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num_labels <= 1: return mask_bin
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask_bin)
    out[labels == largest_label] = 255
    return out

def remove_internal_holes_except_largest(mask_bin: np.ndarray) -> np.ndarray:
    h, w = mask_bin.shape[:2]
    inv = 255 - mask_bin
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if num_labels <= 1: return mask_bin
    border_labels = {labels[0,i] for i in range(w)} | {labels[h-1,i] for i in range(w)} | \
                    {labels[i,0] for i in range(h)} | {labels[i,w-1] for i in range(h)}
    internal = [(lab, stats[lab, cv2.CC_STAT_AREA]) for lab in range(1, num_labels) if lab not in border_labels]
    if not internal: return mask_bin
    keep_lab = max(internal, key=lambda x: x[1])[0]
    out = mask_bin.copy()
    for lab, _ in internal:
        if lab != keep_lab: out[labels == lab] = 255
    return out

def resample_contour(cnt: np.ndarray, n_points: int = 512) -> np.ndarray:
    # 안전장치: cnt가 None이거나 구조가 올바르지 않으면 빈 배열 반환
    if cnt is None or len(cnt) < 2:
        return np.array([])
    
    pts = cnt[:, 0, :].astype(np.float32)
    if not np.allclose(pts[0], pts[-1]): pts = np.vstack([pts, pts[0]])
    d = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] < 1e-6: return pts.reshape((-1, 1, 2))
    
    target = np.linspace(0, s[-1], n_points, endpoint=False)
    x = np.interp(target, s, pts[:, 0])
    y = np.interp(target, s, pts[:, 1])
    return np.stack([x, y], axis=1)

def smooth_contour_fft(cnt: np.ndarray, keep_harmonics: int = 60, n_points: int = 512) -> np.ndarray:
    pts = resample_contour(cnt, n_points=n_points)
    if len(pts) == 0: return None
    
    z = pts[:, 0] + 1j * pts[:, 1]
    Z = np.fft.fft(z)
    K = int(np.clip(keep_harmonics, 3, len(Z) // 2 - 1))
    Z_f = np.zeros_like(Z)
    Z_f[:K], Z_f[-K+1:] = Z[:K], Z[-K+1:]
    z_s = np.fft.ifft(Z_f)
    return np.round(np.stack([z_s.real, z_s.imag], axis=1)).astype(np.int32).reshape((-1, 1, 2))

def get_outer_and_inner_contours(mask_bin: np.ndarray):
    contours, hierarchy = cv2.findContours(mask_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or len(contours) == 0: return None, None
    hierarchy = hierarchy[0]
    
    outer_candidates = [i for i in range(len(contours)) if hierarchy[i][3] == -1]
    if not outer_candidates: return None, None
    
    outer_idx = max(outer_candidates, key=lambda i: cv2.contourArea(contours[i]))
    
    inner_candidates = []
    child = hierarchy[outer_idx][2]
    while child != -1:
        inner_candidates.append(child)
        child = hierarchy[child][0]
    
    inner = None
    if inner_candidates:
        inner_idx = max(inner_candidates, key=lambda i: cv2.contourArea(contours[i]))
        inner = contours[inner_idx]
        
    return contours[outer_idx], inner

# -----------------------------------------------------------------------------
# 2. 메인 로직
# -----------------------------------------------------------------------------
def create_white_shape_png():
    img_path = 'data/250825_174651_A35W_4-3 [16].bmp'
    img = cv2.imread(img_path)
    if img is None: return print("이미지 로드 실패")

    # 영역 추출 및 전처리
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 45, 147]), np.array([11, 183, 255]))
    mask[:, :235] = 0
    mask_main = keep_largest_component(mask)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_main = cv2.morphologyEx(mask_main, cv2.MORPH_CLOSE, k3, iterations=1)

    outer, inner = get_outer_and_inner_contours(mask_main)
    if outer is None: return print("바깥 컨투어 추출 실패")

    # FFT 스무딩 (안쪽 컨투어는 없을 수도 있으므로 체크 필요)
    outer_s = smooth_contour_fft(outer, keep_harmonics=60)
    inner_s = smooth_contour_fft(inner, keep_harmonics=60) if inner is not None else None

    # 마스크 재구성 (확장/축소 로직 포함)
    outer_mask = np.zeros_like(mask_main)
    if outer_s is not None:
        cv2.drawContours(outer_mask, [outer_s], -1, 255, thickness=cv2.FILLED)
        outer_mask = cv2.dilate(outer_mask, k3, iterations=3)

    inner_mask = np.zeros_like(mask_main)
    if inner_s is not None:
        cv2.drawContours(inner_mask, [inner_s], -1, 255, thickness=cv2.FILLED)
        inner_mask = cv2.erode(inner_mask, k3, iterations=3)

    final_mask = outer_mask.copy()
    final_mask[inner_mask > 0] = 0
    final_mask = remove_internal_holes_except_largest(final_mask)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, k3, iterations=1)

    # ✅ 투명 배경에 흰색 코일 이미지만 생성
    white_base = np.full((img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)
    b, g, r = cv2.split(white_base)
    white_shape_rgba = cv2.merge([b, g, r, final_mask])

    # 저장
    os.makedirs('output', exist_ok=True)
    cv2.imwrite('output/roi_coil_C.png', white_shape_rgba)
    print("✅ 완료: output/roi_coil_C.png")

if __name__ == "__main__":
    create_white_shape_png()