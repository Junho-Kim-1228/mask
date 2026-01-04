import cv2
import numpy as np
import os

def create_single_mask():
    img_path = 'data/250825_152739_A35W_2-3 [16].bmp'
    img = cv2.imread(img_path)

    if img is None:
        print("이미지 로드 실패")
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_copper = np.array([0, 45, 147])
    upper_copper = np.array([11, 183, 255])
    mask_color = cv2.inRange(hsv, lower_copper, upper_copper)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_open = cv2.morphologyEx(mask_color, cv2.MORPH_OPEN, kernel_open)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask_closed = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_closed, 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1

    mask_main = np.zeros_like(mask_closed)
    mask_main[labels == largest_label] = 255

    # -------------------------------------------------
    # ✅ 핵심: 컨투어 계층 구조로 내부 유지
    # -------------------------------------------------
    contours, hierarchy = cv2.findContours(
        mask_main, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None:
        print("컨투어 없음")
        return

    hierarchy = hierarchy[0]

    # 가장 큰 외곽 컨투어
    outer_idx = max(
        range(len(contours)),
        key=lambda i: cv2.contourArea(contours[i])
    )

    # 외곽 컨투어를 부모로 갖는 내부 hole들
    hole_indices = [
        i for i, h in enumerate(hierarchy)
        if h[3] == outer_idx
    ]

    # hole 중 가장 큰 것만 유지 (코일 중앙)
    keep_hole_idx = None
    if hole_indices:
        keep_hole_idx = max(
            hole_indices,
            key=lambda i: cv2.contourArea(contours[i])
        )

    final_mask = np.zeros_like(mask_main)

    # 외곽 그리기
    cv2.drawContours(final_mask, [contours[outer_idx]], -1, 255, thickness=cv2.FILLED)

    # 중앙 hole 다시 뚫기
    if keep_hole_idx is not None:
        cv2.drawContours(final_mask, [contours[keep_hole_idx]], -1, 0, thickness=cv2.FILLED)

    # 외곽만 살짝 매끄럽게
    kernel_refine = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_refine)

    os.makedirs('output', exist_ok=True)
    cv2.imwrite('output/test_single_mask_final.png', final_mask)

    print("✅ 내부 유지 + 점 제거 + 외곽 매끄럽게 완료")

if __name__ == "__main__":
    create_single_mask()