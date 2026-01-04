import cv2
import numpy as np
import os

def create_single_mask():
    # 1. 테스트할 이미지 경로 (트랙바에서 썼던 파일명과 동일하게 맞추세요)
    img_path = 'data/250825_152739_A35W_2-3 [16].bmp'
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"이미지를 찾을 수 없습니다: {img_path}")
        return

    # 2. HSV 변환 (노이즈 제거 등 다른 처리 일절 없음)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 3. 지정하신 수치 적용
    lower_copper = np.array([0, 45, 147])
    upper_copper = np.array([11, 183, 255])
    
    # 4. 마스크 생성
    mask = cv2.inRange(hsv, lower_copper, upper_copper)
    
    # 5. 결과 저장 (output 폴더)
    output_folder = 'output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    output_path = os.path.join(output_folder, 'test_single_mask.png')
    cv2.imwrite(output_path, mask)
    
    print(f"단일 이미지 테스트 완료: {output_path}")

if __name__ == "__main__":
    create_single_mask()