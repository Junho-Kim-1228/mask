import cv2
import numpy as np
import os

# 1. 샘플 이미지 로드 (파일 경로가 다르면 수정하세요)
img_path = 'data/250825_152739_A35W_2-3 [16].bmp'
img = cv2.imread(img_path)

if img is None:
    print(f"이미지를 찾을 수 없습니다: {img_path}")
    exit()

def nothing(x):
    pass

# 윈도우 생성
cv2.namedWindow('Trackbar')

# 트랙바 생성 및 아까 찾은 수치로 초기값 설정
cv2.createTrackbar('L-H', 'Trackbar', 0, 179, nothing)
cv2.createTrackbar('L-S', 'Trackbar', 45, 255, nothing)
cv2.createTrackbar('L-V', 'Trackbar', 147, 255, nothing)
cv2.createTrackbar('U-H', 'Trackbar', 11, 179, nothing)
cv2.createTrackbar('U-S', 'Trackbar', 183, 255, nothing)
cv2.createTrackbar('U-V', 'Trackbar', 255, 255, nothing)

while True:
    # 원본 이미지를 HSV로 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 트랙바에서 현재 설정된 값 읽기
    l_h = cv2.getTrackbarPos('L-H', 'Trackbar')
    l_s = cv2.getTrackbarPos('L-S', 'Trackbar')
    l_v = cv2.getTrackbarPos('L-V', 'Trackbar')
    u_h = cv2.getTrackbarPos('U-H', 'Trackbar')
    u_s = cv2.getTrackbarPos('U-S', 'Trackbar')
    u_v = cv2.getTrackbarPos('U-V', 'Trackbar')

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])

    # 설정된 범위로 마스크 생성
    mask = cv2.inRange(hsv, lower, upper)
    
    # 원본에 마스크를 씌운 결과물
    result = cv2.bitwise_and(img, img, mask=mask)

    # 화면 표시 (이미지가 크면 적당히 조절해서 보여줌)
    cv2.imshow('Mask', cv2.resize(mask, (800, 600)))
    cv2.imshow('Result (Coil Only)', cv2.resize(result, (800, 600)))

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        print(f"최종 수치: lower=[{l_h}, {l_s}, {l_v}], upper=[{u_h}, {u_s}, {u_v}]")
        break

cv2.destroyAllWindows()