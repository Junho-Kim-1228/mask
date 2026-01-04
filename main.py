import cv2
import numpy as np
import glob
import os

def create_coil_mask(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    
    # HSV 변환 및 코일 색상 추출
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_copper = np.array([5, 40, 40])
    upper_copper = np.array([25, 255, 255])
    
    mask = cv2.inRange(hsv, lower_copper, upper_copper)
    
    # 노이즈 제거 (Closing 연산)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

# 1. 'data' 폴더 내의 모든 .bmp 파일을 찾습니다.
image_files = glob.glob('data/*.bmp')
print(f"찾은 이미지 개수: {len(image_files)}개")

final_mask = None

# 2. 모든 마스크의 합집합 구하기
for f in image_files:
    print(f"처리 중: {f}")
    m = create_coil_mask(f)
    if m is not None:
        if final_mask is None:
            final_mask = m
        else:
            final_mask = cv2.bitwise_or(final_mask, m)

# 3. 'output' 폴더에 저장하기
if final_mask is not None:
    # output 폴더가 없으면 새로 만듭니다.
    output_folder = 'output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"'{output_folder}' 폴더를 생성했습니다.")

    # 파일 저장 경로 설정
    output_path = os.path.join(output_folder, 'final_union_mask.png')
    cv2.imwrite(output_path, final_mask)
    
    print("-" * 30)
    print(f"성공! 결과물이 다음 경로에 저장되었습니다: {output_path}")
else:
    print("오류: 이미지를 처리하지 못했습니다. 'data' 폴더와 파일을 확인해 주세요.")