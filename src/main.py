import cv2
import numpy as np
import glob

def create_coil_mask(image_path):
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # 1. HSV 색공간으로 변환 (구리색/주황색 계열 추출에 용이)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 2. 코일의 구리색 범위 설정 (이미지 밝기에 따라 조절 필요)
    # 일반적인 구리색/밝은 주황색 범위
    lower_copper = np.array([5, 50, 50])
    upper_copper = np.array([25, 255, 255])
    
    # 3. 마스크 생성 (코일 부분만 흰색)
    mask = cv2.inRange(hsv, lower_copper, upper_copper)
    
    # 4. 형태학적 연산으로 미세 노이즈 제거 및 코일 내부 채우기
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # 내부 구멍 메우기
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 외곽 노이즈 제거
    
    return mask

# 이미지 파일 리스트 (업로드하신 파일 경로들)
image_files = glob.glob('*.bmp') 

final_mask = None

for file in image_files:
    current_mask = create_coil_mask(file)
    
    if current_mask is not None:
        if final_mask is None:
            final_mask = current_mask
        else:
            # 합집합 연산 (OR)
            final_mask = cv2.bitwise_or(final_mask, current_mask)

# 결과 저장 및 확인
cv2.imwrite('final_union_mask.png', final_mask)
print("최종 합집합 마스크 생성 완료")