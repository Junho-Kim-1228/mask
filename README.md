# mask_project

프로젝트 전체 개요, 시도한 방법, 파일 역할, 전체 사용 흐름을 정리한 메인 문서다.

PCB 원본 이미지에서 **코일 영역만 분리**하기 위한 프로젝트다.  
출발점은 OpenCV 규칙 기반 마스크 생성기였고, 현재는 **U-Net++ + EfficientNet-B4** 기반 AI segmentation 파이프라인까지 포함한다.

이 프로젝트의 핵심 목표는 단순 배경 제거가 아니다.

- 코일만 남길 것
- 다른 부품, 회색/밝은 부품, 배경은 제거할 것
- 코일 내부 홀은 유지할 것
- 경계선 불량, 찍힘, 풀림 같은 실제 형상 정보는 최대한 보존할 것

## 1. 현재 결론

현재 기준으로 프로젝트에서 실제로 쓰는 방향은 아래다.

- 초기 라벨 초안은 규칙 기반 baseline으로 만든다
- CVAT에서 사람이 수정한 데이터셋을 기준으로 AI segmentation 모델을 반복 학습한다
- 학습된 모델로 raw 이미지를 프리라벨링하거나, 코일만 남긴 masked image를 만든다
- 완벽한 segmentation 미관보다 **불량 보존**을 우선한다

현재 실험상 가장 많이 쓰는 추론 설정은 다음과 같다.

- 모델: `models/coil_unetpp_effb4_scratch_v8_best.pt`
- 입력 크기: `512`
- `mask-threshold`: `0.30`
- `min-component-area`: `64`
- `outer-recover-kernel`: `0`

이 설정을 쓰는 이유는 다음과 같다.

- 경계를 너무 엄격하게 자르면 실제 결함이 같이 사라진다
- threshold `0.50`보다 `0.30`에서 검정 불량과 경계가 더 잘 살아남았다
- 반대로 recover나 shift 같은 후처리 트릭은 근본 해결이 아니었다

## 2. 프로젝트 요약

### 문제 정의

- 입력: PCB 원본 이미지
- 출력 1: CVAT용 segmentation dataset
- 출력 2: 코일만 남긴 masked image
- 목적: 라벨링 효율화, segmentation 모델 학습, 대량 프리라벨링, 후속 anomaly 검사용 입력 생성

### 현재 접근법

프로젝트는 아래 3단계로 운영한다.

1. 규칙 기반 baseline으로 초기 마스크 초안 확보
2. CVAT 수정본으로 AI segmentation 모델 학습
3. 학습된 모델로 대량 프리라벨링 후 다시 CVAT 수정, 재학습 반복

### 왜 이렇게 구성했는가

처음부터 전부 수작업 라벨링하면 비용이 너무 크다.  
그래서

- baseline 초안
- 소량 수작업 수정
- 1차 모델 학습
- 대량 프리라벨
- 추가 수정

의 반복형 구조로 바꿨다.

## 3. 지금까지 실제로 시도한 방법

### 3-1. 규칙 기반 baseline

초기에는 `src/process_all.py`를 사용했다.

핵심 아이디어:

- edge / texture 기반 후보 추출
- morphology
- contour
- adaptive color gate
- 내부 홀 유지

장점:

- 라벨링 초안 생성 속도가 빠름
- 데이터셋이 전혀 없을 때 출발점으로 유용함

한계:

- 이미지별 색/조명 변화에 취약
- 경계가 조금씩 깎이거나 남는 편향이 생김
- 검정색 결함이나 색이 다른 불량에 약함

### 3-2. 1차 AI 모델

초기 AI 모델은 `U-Net++ + EfficientNet-B4`로 시작했다.

기본 방향:

- binary segmentation
- `background / coil`
- supervised learning

초기에는 `BCE + Dice` 계열로 학습했지만, 다음 문제가 남았다.

- 경계를 조금 덜 잡아도 metric이 크게 안 떨어짐
- 검정색 찍힘/불량은 잘 놓침
- 전체 dice는 높아도 실제 불량 보존이 아쉬움

### 3-3. 경계 보존용 보강셋

경계가 잘리는 문제를 줄이기 위해 `3.2ds`를 따로 만들었다.

의도:

- 풀림, 얇은 경계, 외곽이 잘리던 샘플 보강

이후 시도:

- 추가 라벨셋 병합
- fine-tune
- scratch 재학습 비교

결론:

- 경계 보강 자체는 의미가 있었지만
- 검정 불량/색 다른 불량까지 자동으로 해결되지는 않았다

### 3-4. 색이 다른 불량 보강셋

검정색 찍힘, 색이 다른 불량을 보강하기 위해 `4.2ds`를 만들었다.

이 데이터셋의 의미:

- segmentation이 일반적인 코일 색만 배우지 않도록
- “이상한 색이지만 코일로 남겨야 하는 부분”을 강하게 보여주는 hard case 세트

### 3-5. Fine-tune vs Scratch

둘 다 실제로 실험했다.

- 기존 best 모델 기준 fine-tune
- 새 체크포인트 이름으로 scratch 재학습

관찰:

- metric만 보면 기존 fine-tune 쪽이 좋을 때도 있었다
- 하지만 시각적으로는 scratch 쪽이 더 자연스러운 경우가 있었다

즉 이 프로젝트는 metric 숫자만으로 모델을 선택하지 않고,

- 경계 보존
- 검정 불량 보존
- 배경 유입 정도

를 같이 본다.

### 3-6. Threshold, recover, shift 실험

경계가 잘리는 문제를 줄이기 위해 아래도 실험했다.

- `mask-threshold` 하향
- `outer-recover-kernel`
- `mask shift`
- refine crop 2단계 추론

현재 판단:

- `mask-threshold` 조정은 실제로 영향이 컸다
- `outer-recover-kernel`은 보조적
- `mask shift`는 디버그용일 뿐, 근본 해결은 아님
- refine crop은 내부적으로만 의미가 있고, 최종 전략의 핵심은 아님

### 3-7. 현재 학습 전략

최근에는 hard case에 더 민감하게 반응하도록 학습 로직을 바꿨다.

현재 반영된 것:

- `BoundaryWeighted BCE + Focal Tversky`
- `4.2ds` oversampling
- `4.2ds` 일부를 hard validation으로 고정
- `focus_val_tversky` 기준 best checkpoint 선택
- 20 epoch마다 중간 checkpoint 저장

즉 지금은 단순 `val_dice`가 아니라,  
실제로 놓치기 쉬운 hard case에서 잘 버티는 모델을 뽑는 방향으로 바뀌었다.

## 4. 현재 폴더 구조

현재 repo에서 실사용 기준으로 중요한 폴더는 아래다.

```text
mask_project/
├─ data/                    # baseline 초기 마스크 생성용 원본 이미지
├─ dataset/                 # 현재 프리라벨 대상 raw 이미지 풀
├─ 1.0ds/                   # 1차 수동 수정 완료 정답셋
├─ 2.1ds/                   # 2차 수정 완료 정답셋
├─ 3.2ds/                   # 경계 보강용 수정 완료 정답셋
├─ 4.2ds/                   # 검정/색 다른 불량 보강용 수정 완료 정답셋
├─ models/
│  ├─ .gitkeep
│  └─ coil_unetpp_effb4_scratch_v8_best.pt
├─ output/
│  └─ coil_only_ai/         # AI masked image 저장 위치
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
│  ├─ apply_ai_mask.py
│  └─ make_cvat_zip.py
├─ README.md
├─ README_AI.md
├─ requirements.txt
└─ requirements_ai.txt
```

## 5. `src/` 파일 역할

### Baseline 관련

- [src/process_all.py](/mnt/c/users/wnsgh/kjhdev/mask_project/src/process_all.py)  
  규칙 기반 초기 마스크 생성기.  
  `data/` 원본을 읽어서 `dataset/SegmentationClass`, `dataset/SegmentationObject`에 CVAT용 마스크를 만든다.

- [src/trackbar.py](/mnt/c/users/wnsgh/kjhdev/mask_project/src/trackbar.py)  
  baseline 마스크 튜닝용 GUI.  
  `trim`, `smooth`, 내부 홀 비율 등 rule-based 파라미터를 손으로 만질 때 사용한다.

### AI 관련

- [src/config_ai.py](/mnt/c/users/wnsgh/kjhdev/mask_project/src/config_ai.py)  
  AI 기본 설정 파일.  
  모델 구조, 기본 경로, threshold, 입력 크기 등 공통 상수를 모아둔다.

- [src/io_utils_ai.py](/mnt/c/users/wnsgh/kjhdev/mask_project/src/io_utils_ai.py)  
  이미지 로드, resize/pad, bbox 계산, crop, mask 적용, 디렉터리 생성 등 공통 유틸.

- [src/segment_model.py](/mnt/c/users/wnsgh/kjhdev/mask_project/src/segment_model.py)  
  U-Net++ + EfficientNet-B4 모델 생성, checkpoint 로드, device 선택, 추론 래퍼.

- [src/postprocess_ai.py](/mnt/c/users/wnsgh/kjhdev/mask_project/src/postprocess_ai.py)  
  probability map 후처리.  
  threshold, small component 제거, morphology, largest component 유지, inner hole 보존 등을 담당한다.

- [src/dataset_ai.py](/mnt/c/users/wnsgh/kjhdev/mask_project/src/dataset_ai.py)  
  PyTorch dataset loader.  
  현재는 색상 generalization을 위해 brightness/contrast/Hue-Saturation/ToGray augmentation도 포함한다.

- [src/train_ai.py](/mnt/c/users/wnsgh/kjhdev/mask_project/src/train_ai.py)  
  학습 엔트리포인트.  
  oversampling, hard validation, boundary-weighted loss, 20 epoch 주기 저장까지 포함한다.

- [src/prepare_cvat_dataset_ai.py](/mnt/c/users/wnsgh/kjhdev/mask_project/src/prepare_cvat_dataset_ai.py)  
  `1.0ds`, `2.1ds`, `3.2ds`, `4.2ds` 같은 완료본을 train/val 구조로 변환한다.  
  여러 source를 merge할 수 있고, hard val source도 지정 가능하다.

- [src/prelabel_cvat_dataset_ai.py](/mnt/c/users/wnsgh/kjhdev/mask_project/src/prelabel_cvat_dataset_ai.py)  
  raw 이미지 폴더를 읽어서 CVAT용 VOC-style segmentation dataset을 생성한다.

- [src/apply_ai_mask.py](/mnt/c/users/wnsgh/kjhdev/mask_project/src/apply_ai_mask.py)  
  raw 이미지에 AI 마스크를 적용해서 코일만 남긴 `masked image`를 저장한다.

- [src/make_cvat_zip.py](/mnt/c/users/wnsgh/kjhdev/mask_project/src/make_cvat_zip.py)  
  VOC-style dataset을 CVAT 업로드용 zip으로 묶는다.

## 6. CVAT용 마스크 형식

현재 CVAT용 마스크는 **1채널 index mask**를 사용한다.

- 배경: `0`
- 코일: `1`

이유:

- 예전 3채널 `(0,0,0)` / `(255,255,255)` 마스크는 CVAT에서  
  `Undeclared color (255, 255, 255)` 오류가 났다
- 1채널 index mask는 CVAT에서 훨씬 안정적으로 읽힌다

즉 이미지 뷰어로 보면 거의 검정처럼 보여도 정상이다.

## 7. 환경 준비

### 가상환경 활성화

```bash
conda activate mask_vision
```

### baseline 의존성

```bash
pip install -r requirements.txt
```

### AI 의존성

```bash
pip install -r requirements_ai.txt
```

노트북 환경에서는 GPU 메모리/발열 때문에 보통 아래를 권장한다.

- `input-size 512`
- `batch-size 1`

## 8. 사용법

### 8-1. 초기 baseline 마스크 생성

`data/`에 원본 이미지를 넣고:

```bash
python src/process_all.py
```

결과:

- `dataset/SegmentationClass/*.png`
- `dataset/SegmentationObject/*.png`
- `dataset/ImageSets/Segmentation/default.txt`

용도:

- 초기 CVAT 수정용 bootstrap mask 생성

### 8-2. CVAT 업로드용 zip 만들기

```bash
python src/make_cvat_zip.py --dataset-dir 1.0ds --output-zip 1.0ds_for_cvat.zip
```

예시:

```bash
python src/make_cvat_zip.py --dataset-dir 4.2ds --output-zip 4.2ds_for_cvat.zip
```

### 8-3. 완료본 데이터셋을 train/val로 변환

단일 source:

```bash
python src/prepare_cvat_dataset_ai.py --source-dir 1.0ds --output-dir prepared_1.0ds --val-ratio 0.2 --overwrite
```

여러 source merge:

```bash
python src/prepare_cvat_dataset_ai.py --source-dir 1.0ds --source-dir 2.1ds --source-dir 3.2ds --source-dir 4.2ds --output-dir prepared_trainset_v8 --val-ratio 0.2 --hard-val-source 4.2ds --hard-val-count 5 --overwrite
```

의미:

- `4.2ds`에서 5장을 hard validation에 강제로 포함

### 8-4. 현재 권장 재학습 명령어

현재 hard case 대응용 학습 설정 예시는 이렇다.

```bash
python src/train_ai.py --train-images-dir prepared_trainset_v8/train/images --train-masks-dir prepared_trainset_v8/train/masks --val-images-dir prepared_trainset_v8/val/images --val-masks-dir prepared_trainset_v8/val/masks --checkpoint-path models/coil_unetpp_effb4_scratch_v8_best.pt --last-checkpoint-path models/coil_unetpp_effb4_scratch_v8_last.pt --input-size 512 --batch-size 1 --epochs 80 --lr 1e-4 --device auto --oversample-source 4.2ds --oversample-factor 16.0 --tversky-alpha 0.20 --tversky-beta 0.80 --focal-gamma 1.5 --boundary-weight 3.0 --focus-val-source 4.2ds --best-metric focus_val_tversky --save-every 20
```

이 명령어의 핵심:

- `4.2ds`를 16배 oversampling
- `4.2ds` 일부를 val에서도 직접 확인
- `focus_val_tversky` 기준으로 best checkpoint 선택
- `20/40/60/80` epoch checkpoint 저장

### 8-5. raw 이미지 프리라벨링

현재 권장 모델/설정:

```bash
python src/prelabel_cvat_dataset_ai.py --input-dir dataset --output-dir 4.3ds --model-path models/coil_unetpp_effb4_scratch_v8_best.pt --device auto --input-size 512 --mask-threshold 0.30 --min-component-area 64 --outer-recover-kernel 0 --overwrite
```

설명:

- `dataset/` 원본 raw 이미지를 읽는다
- `4.3ds/`에 CVAT 업로드 가능한 segmentation dataset을 만든다
- 현재 실험상 `mask-threshold 0.30`이 검정 불량과 경계를 제일 잘 살렸다

### 8-6. 코일만 남긴 masked image 생성

```bash
python src/apply_ai_mask.py --input-dir dataset --output-dir output/coil_only_ai --model-path models/coil_unetpp_effb4_scratch_v8_best.pt --device auto --input-size 512 --mask-threshold 0.30 --min-component-area 64 --outer-recover-kernel 0 --overwrite
```

이 결과는 보통 이렇게 부른다.

- `masked image`
- `foreground-only image`
- `mask applied result`

즉 라벨링용 mask PNG와는 다르고,  
원본에서 코일만 남기고 나머지를 검정으로 날린 전처리 결과다.

## 9. 모델 선택 기준

이 프로젝트에서는 모델을 숫자 하나만 보고 고르지 않는다.

같이 보는 기준:

- `val_dice`
- `val_tversky`
- hard case(`4.2ds`)에서의 `focus_val_tversky`
- 실제 CVAT 프리라벨 결과
- 경계가 과하게 잘리지 않는지
- 검정 불량을 살리는지
- 배경이 과하게 들어오지 않는지

즉 metric은 참고용이고, 최종 판단은 실제 샘플 비교까지 포함한다.

## 10. 현재 프로젝트에서 배운 점

이 프로젝트에서 중요했던 건 단순히 segmentation 점수를 높이는 게 아니었다.

실제로는 다음이 더 중요했다.

- 경계를 조금 덜 잡아도 metric은 잘 나올 수 있다
- 검정색 불량은 segmentation이 그냥 배경으로 무시해버릴 수 있다
- 라벨 보강셋(`3.2ds`, `4.2ds`)은 실제 모델 편향을 교정하는 데 매우 중요하다
- 후처리 트릭보다, hard case를 학습과 validation에 직접 반영하는 게 더 효과적이다

즉 이 프로젝트는 단순한 segmentation 예제가 아니라,

- 라벨링 반복
- hard case 수집
- metric과 실제 샘플 비교
- 모델 선택 기준 재설계

까지 포함한 **현실적인 비전 데이터 엔지니어링/모델링 프로젝트**에 가깝다.
