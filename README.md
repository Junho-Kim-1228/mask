# mask_project

PCB 이미지에서 **코일 영역만 정확하게 분리**하기 위해 만든 프로젝트다.  
초기에는 OpenCV 기반 규칙형 전처리로 시작했고, 이후 **U-Net++ + EfficientNet-B4** 기반 AI segmentation 파이프라인으로 확장했다.

이 프로젝트의 핵심은 단순 배경 제거가 아니라 다음 조건을 만족하는 것이다.

- 코일만 남길 것
- 다른 부품, 밝은 부품, 회색 부품은 제거할 것
- 코일 내부 홀은 유지할 것
- 홀 안쪽으로 튀어나온 실제 코일 형상은 최대한 보존할 것

---

## 1. 프로젝트 개요

### 문제 정의
- 입력: PCB 원본 이미지
- 출력: 코일만 분리된 마스크 또는 코일 전용 annotation dataset
- 목적: CVAT 기반 라벨링 효율화 + AI segmentation 모델 학습 + 대량 프리라벨링

### 현재 구조
- **Baseline 단계**: 규칙 기반 마스크 생성으로 초기 라벨링 초안 확보
- **AI 단계**: 사람이 수정한 정답셋으로 U-Net++ 모델 학습
- **반복 단계**: 학습된 모델로 대량 프리라벨링 -> CVAT 수정 -> 재학습

### 최근 학습 결과
- 2차 학습(`1.0ds + 2.1ds`) 기준
- `best_val_dice = 0.9852`

이 수치는 최종 정답 품질을 보장하는 값은 아니지만, **대량 프리라벨링용 모델로는 충분히 실용적인 상태**라는 판단 기준으로 사용했다.

---

## 2. 전체 프로세스

이 프로젝트는 아래 순서로 운영했다.

### 단계 A. 규칙 기반 Baseline으로 초기 마스크 초안 생성
1. 원본 이미지를 `data/`에 넣는다.
2. `src/process_all.py`로 코일 마스크를 생성한다.
3. 결과를 CVAT가 읽을 수 있는 VOC-style segmentation 구조로 저장한다.
4. 이를 CVAT에서 열어 사람이 직접 수정한다.
5. 수정 완료본을 `1.0ds/`로 보관한다.

### 단계 B. 1차 AI 학습
1. `1.0ds/`를 train/val 구조로 나눈다.
2. `src/train_ai.py`로 U-Net++ 모델을 학습한다.
3. 학습된 모델(`models/coil_unetpp_effb4_best.pt`)을 사용해 raw 이미지 50장을 프리라벨링한다.
4. CVAT에서 수정 완료본을 `2.1ds/`로 저장한다.

### 단계 C. 2차 AI 학습
1. `1.0ds/`와 `2.1ds/`를 합쳐서 다시 train/val split을 만든다.
2. 같은 모델을 다시 학습한다.
3. 개선된 best 모델로 대량 raw 이미지 세트를 프리라벨링한다.
4. 현재는 `3.0ds/` 형태로 결과를 생성해서 다시 CVAT 수정에 투입할 수 있다.

즉 이 프로젝트는 **한 번에 정답을 만드는 방식이 아니라, 라벨링과 모델 학습을 반복하면서 품질을 끌어올리는 구조**다.

---

## 3. 디렉터리 구조

현재 repo에서 자주 쓰는 주요 폴더는 아래와 같다.

```text
mask_project/
├─ data/                    # 초기 baseline 마스크 생성을 위한 원본 이미지
├─ dataset/                 # 현재 프리라벨 대상 raw 이미지 풀
├─ 1.0ds/                   # CVAT에서 수정 완료한 1차 정답셋
├─ 2.1ds/                   # 2차 수정 완료 정답셋
├─ 3.0ds/                   # 현재 best 모델로 프리라벨링한 대량 결과셋
├─ prepared_1.0ds/          # 1.0ds를 train/val로 나눈 학습용 구조
├─ prepared_trainset/       # 1.0ds + 2.1ds를 합친 재학습용 구조
├─ models/
│  ├─ coil_unetpp_effb4_best.pt
│  └─ coil_unetpp_effb4_last.pt
├─ output/
│  └─ coil_only/            # baseline이 코일만 남긴 BMP 결과를 저장하는 폴더
├─ src/
│  ├─ process_all.py
│  ├─ trackbar.py
│  ├─ config_ai.py
│  ├─ io_utils_ai.py
│  ├─ segment_model.py
│  ├─ postprocess_ai.py
│  ├─ dataset_ai.py
│  ├─ train_ai.py
│  ├─ prepare_cvat_dataset_ai.py
│  ├─ prelabel_cvat_dataset_ai.py
│  └─ make_cvat_zip.py
├─ README.md
├─ README_AI.md
├─ requirements.txt
└─ requirements_ai.txt
```

주의:
- 이 프로젝트에서는 `dataset/` 폴더를 **두 단계에서 다른 용도**로 썼다.
- 초기 bootstrap 단계에서는 `src/process_all.py`가 `dataset/` 아래에 CVAT용 segmentation 구조를 만들었다.
- 그 결과를 `1.0ds/`로 정리한 뒤에는, `dataset/`를 다시 **raw 이미지 풀**로 재사용했다.

---

## 4. 파일별 역할

### Baseline 관련
- [`src/process_all.py`](/mnt/c/Users/wnsgh/kjhdev/mask_project/src/process_all.py)  
  규칙 기반 코일 마스크 생성기.  
  에지/텍스처, morphology, contour, adaptive color gate를 이용해 초기 마스크를 만든다.

- [`src/trackbar.py`](/mnt/c/Users/wnsgh/kjhdev/mask_project/src/trackbar.py)  
  baseline 파이프라인 튜닝용 GUI.  
  `trim`, `smooth`, `내부W/H` 같은 값을 손으로 조정하면서 baseline 결과를 빠르게 확인할 때 사용한다.

### AI 관련
- [`src/config_ai.py`](/mnt/c/Users/wnsgh/kjhdev/mask_project/src/config_ai.py)  
  모델 구조, 입력 크기, threshold, 경로 같은 AI 기본 설정을 모아둔 파일.

- [`src/io_utils_ai.py`](/mnt/c/Users/wnsgh/kjhdev/mask_project/src/io_utils_ai.py)  
  이미지 로드, resize/pad, 디렉터리 생성, 파일 탐색 같은 공통 입출력 유틸.

- [`src/segment_model.py`](/mnt/c/Users/wnsgh/kjhdev/mask_project/src/segment_model.py)  
  U-Net++ + EfficientNet-B4 모델 생성, weight 로드, device 선택, 추론 래퍼.

- [`src/postprocess_ai.py`](/mnt/c/Users/wnsgh/kjhdev/mask_project/src/postprocess_ai.py)  
  probability map을 threshold해서 binary mask로 바꾸고,  
  small blob 제거, conservative morphology, largest component 유지 같은 최소 후처리를 담당한다.

- [`src/dataset_ai.py`](/mnt/c/Users/wnsgh/kjhdev/mask_project/src/dataset_ai.py)  
  PyTorch 학습용 dataset loader.  
  이미지와 마스크 쌍을 읽고 transform을 적용한다.

- [`src/train_ai.py`](/mnt/c/Users/wnsgh/kjhdev/mask_project/src/train_ai.py)  
  학습 엔트리포인트.  
  `BCE + Dice` loss 기반으로 binary segmentation 모델을 학습한다.  
  `--resume-from`으로 중단된 학습도 이어갈 수 있다.

- [`src/prepare_cvat_dataset_ai.py`](/mnt/c/Users/wnsgh/kjhdev/mask_project/src/prepare_cvat_dataset_ai.py)  
  `1.0ds`, `2.1ds` 같은 CVAT 완료본을 train/val 구조로 변환한다.  
  여러 `--source-dir`를 동시에 받아서 dataset merge도 가능하다.

- [`src/prelabel_cvat_dataset_ai.py`](/mnt/c/Users/wnsgh/kjhdev/mask_project/src/prelabel_cvat_dataset_ai.py)  
  학습된 모델로 raw 이미지 폴더를 읽어서,  
  CVAT에 바로 import 가능한 VOC-style segmentation dataset을 생성한다.

- [`src/make_cvat_zip.py`](/mnt/c/Users/wnsgh/kjhdev/mask_project/src/make_cvat_zip.py)  
  `labelmap.txt`, `ImageSets/Segmentation/default.txt`, `SegmentationClass`, `SegmentationObject` 구조를 zip으로 묶어 CVAT 업로드 파일을 만든다.

---

## 5. 마스크 형식과 CVAT 호환성

이 프로젝트에서 CVAT 업로드용 segmentation mask는 **1채널 index mask**를 사용한다.

- 배경: `0`
- 코일: `1`

이 형식을 쓰는 이유는, 초기에 3채널 `(0,0,0)` / `(255,255,255)` 마스크를 사용했을 때 CVAT에서  
`Undeclared color (255, 255, 255)` 오류가 발생했기 때문이다.

즉 지금은:
- 사람이 눈으로 보면 거의 검정처럼 보일 수 있지만
- CVAT용 annotation dataset으로는 더 안정적인 형식이다.

---

## 6. 환경 설정

### Conda 가상환경 활성화
```bash
conda activate mask_vision
```

### Baseline 의존성 설치
```bash
pip install -r requirements.txt
```

### AI 의존성 설치
```bash
pip install -r requirements_ai.txt
```

노트북 환경이라면 GPU 메모리와 발열 문제 때문에 학습 시 `input-size 512`, `batch-size 1`부터 시작하는 편이 안정적이다.

---

## 7. 사용 방법

### 7-1. 초기 baseline 마스크 생성

`data/`에 원본 이미지를 넣고 실행:

```bash
python src/process_all.py
```

이 스크립트는 다음 용도로 사용한다.
- baseline 규칙 기반 마스크 생성
- 초기 CVAT 수정용 초안 확보
- 코일만 남긴 BMP 결과 저장
- CVAT용 index mask를 `dataset/SegmentationClass`, `dataset/SegmentationObject`에 저장

주의:
- 이 단계는 **초기 bootstrap용**이다.
- 이후 AI 반복 단계에서는 주로 `1.0ds`, `2.1ds`, `dataset/`, `3.0ds` 흐름을 사용한다.

### 7-2. CVAT 업로드용 zip 생성

VOC-style dataset 폴더를 zip으로 묶을 때:

```bash
python src/make_cvat_zip.py --dataset-dir 1.0ds --output-zip 1.0ds_for_cvat.zip
```

또는:

```bash
python src/make_cvat_zip.py --dataset-dir 3.0ds --output-zip 3.0ds_for_cvat.zip
```

---

## 8. 학습 파이프라인

### 8-1. 1차 라벨링 데이터 준비

`1.0ds/`를 train/val 구조로 변환:

```bash
python src/prepare_cvat_dataset_ai.py --source-dir 1.0ds --output-dir prepared_1.0ds --val-ratio 0.2 --overwrite
```

생성 결과:

```text
prepared_1.0ds/
├─ train/
│  ├─ images/
│  └─ masks/
└─ val/
   ├─ images/
   └─ masks/
```

### 8-2. 1차 모델 학습

```bash
python src/train_ai.py --train-images-dir prepared_1.0ds/train/images --train-masks-dir prepared_1.0ds/train/masks --val-images-dir prepared_1.0ds/val/images --val-masks-dir prepared_1.0ds/val/masks --checkpoint-path models/coil_unetpp_effb4_best.pt --last-checkpoint-path models/coil_unetpp_effb4_last.pt --input-size 512 --batch-size 1 --epochs 40 --device auto
```

### 8-3. 학습 중단 후 이어서 학습

```bash
python src/train_ai.py --train-images-dir prepared_1.0ds/train/images --train-masks-dir prepared_1.0ds/train/masks --val-images-dir prepared_1.0ds/val/images --val-masks-dir prepared_1.0ds/val/masks --checkpoint-path models/coil_unetpp_effb4_best.pt --last-checkpoint-path models/coil_unetpp_effb4_last.pt --resume-from models/coil_unetpp_effb4_last.pt --input-size 512 --batch-size 1 --epochs 40 --device auto
```

### 8-4. 2차 학습용 dataset merge

`1.0ds`와 `2.1ds`를 합쳐서 재학습용 split 생성:

```bash
python src/prepare_cvat_dataset_ai.py --source-dir 1.0ds --source-dir 2.1ds --output-dir prepared_trainset --val-ratio 0.2 --overwrite
```

### 8-5. 2차 모델 학습

```bash
python src/train_ai.py --train-images-dir prepared_trainset/train/images --train-masks-dir prepared_trainset/train/masks --val-images-dir prepared_trainset/val/images --val-masks-dir prepared_trainset/val/masks --checkpoint-path models/coil_unetpp_effb4_best.pt --last-checkpoint-path models/coil_unetpp_effb4_last.pt --input-size 512 --batch-size 1 --epochs 40 --device auto
```

---

## 9. 대량 프리라벨링

### 9-1. raw 이미지 50장 프리라벨링

`dataset/`에 raw 이미지 50장을 넣은 뒤:

```bash
python src/prelabel_cvat_dataset_ai.py --input-dir dataset --output-dir 2.0ds --model-path models/coil_unetpp_effb4_best.pt --device auto --input-size 512
```

이후 CVAT에서 수정 완료한 결과를 `2.1ds/`로 관리한다.

### 9-2. 대량 raw 이미지 프리라벨링

현재 best 모델로 raw 이미지 대량 세트를 프리라벨링:

```bash
python src/prelabel_cvat_dataset_ai.py --input-dir dataset --output-dir 3.0ds --model-path models/coil_unetpp_effb4_best.pt --device auto --input-size 512 --overwrite
```

생성 결과:

```text
3.0ds/
├─ labelmap.txt
├─ JPEGImages/
├─ SegmentationClass/
├─ SegmentationObject/
└─ ImageSets/Segmentation/default.txt
```

CVAT 업로드 zip 생성:

```bash
python src/make_cvat_zip.py --dataset-dir 3.0ds --output-zip 3.0ds_for_cvat.zip
```

---

## 10. 이 프로젝트에서 중요한 설계 원칙

- baseline과 AI 파이프라인을 섞지 않는다.
- 초기 라벨링 비용을 줄이기 위해 baseline을 **bootstrap 도구**로 사용한다.
- AI 모델은 바로 최종 정답을 만드는 용도가 아니라 **프리라벨링 가속기**로 사용한다.
- 후처리는 최소화하고, 실제 코일 형상 보존을 우선한다.
- 내부 홀은 유지한다.
- morphology로 억지로 예쁜 모양을 만드는 것을 피한다.

---

## 11. 포트폴리오 관점에서의 의미

이 프로젝트는 단순한 이미지 전처리 스크립트가 아니라, 실제 현업형 annotation workflow를 설계하고 구현한 사례다.

핵심 포인트:
- 규칙 기반 baseline 구축
- CVAT 호환 segmentation dataset 구조 설계
- 1채널 index mask 기반 import 오류 해결
- U-Net++ + EfficientNet-B4 기반 binary segmentation 학습 파이프라인 구축
- 소량 수작업 라벨 -> AI 프리라벨 -> 사람 수정 -> 재학습의 **iterative labeling loop** 구현
- 노트북 GPU 환경에서 안정적으로 돌릴 수 있도록 `input-size`, `batch-size`, resume 전략까지 정리

즉 이 프로젝트의 가치는 단순히 “마스크를 잘 뽑았다”가 아니라,  
**라벨링 비용을 줄이고 데이터셋을 점진적으로 확장할 수 있는 운영형 파이프라인을 만들었다는 점**에 있다.

---

## 12. 주의사항

- `input-size 768`, `batch-size 2`는 노트북 GPU 환경에서 무거울 수 있다.
- 학습 중 노트북 전체가 꺼진다면, 파이썬 에러보다 **발열 / 전원 / 드라이버 리셋** 문제일 가능성이 크다.
- 이 경우 아래 설정이 더 안정적이다:

```bash
--input-size 512 --batch-size 1
```

- CVAT용 index mask는 일반 이미지 뷰어에서 거의 검정처럼 보일 수 있다.  
  이것은 오류가 아니라 **0/1 index mask의 정상적인 표시 방식**이다.
