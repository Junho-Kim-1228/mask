# README_AI

이 문서는 AI segmentation 파이프라인만 빠르게 다시 볼 때 사용하는 보조 문서다.  
프로젝트 전체 흐름, baseline bootstrap, CVAT workflow, 파일 역할, 포트폴리오 설명은 [`README.md`](/mnt/c/Users/wnsgh/kjhdev/mask_project/README.md)를 기준으로 본다.

## 핵심 요약

- 모델: `U-Net++ + EfficientNet-B4`
- 문제 정의: `binary segmentation (background / coil)`
- 주 용도: raw PCB 이미지에 대한 **프리라벨링**
- 출력 형식: CVAT import 가능한 VOC-style dataset

## 자주 쓰는 명령어

### 1. 1차 라벨셋을 train/val로 분리
```bash
python src/prepare_cvat_dataset_ai.py --source-dir 1.0ds --output-dir prepared_1.0ds --val-ratio 0.2 --overwrite
```

### 2. 1차 학습
```bash
python src/train_ai.py --train-images-dir prepared_1.0ds/train/images --train-masks-dir prepared_1.0ds/train/masks --val-images-dir prepared_1.0ds/val/images --val-masks-dir prepared_1.0ds/val/masks --checkpoint-path models/coil_unetpp_effb4_best.pt --last-checkpoint-path models/coil_unetpp_effb4_last.pt --input-size 512 --batch-size 1 --epochs 40 --device auto
```

### 3. 2차 학습용 merge
```bash
python src/prepare_cvat_dataset_ai.py --source-dir 1.0ds --source-dir 2.1ds --output-dir prepared_trainset --val-ratio 0.2 --overwrite
```

### 4. 2차 학습
```bash
python src/train_ai.py --train-images-dir prepared_trainset/train/images --train-masks-dir prepared_trainset/train/masks --val-images-dir prepared_trainset/val/images --val-masks-dir prepared_trainset/val/masks --checkpoint-path models/coil_unetpp_effb4_best.pt --last-checkpoint-path models/coil_unetpp_effb4_last.pt --input-size 512 --batch-size 1 --epochs 40 --device auto
```

### 5. 대량 프리라벨링
```bash
python src/prelabel_cvat_dataset_ai.py --input-dir dataset --output-dir 3.0ds --model-path models/coil_unetpp_effb4_best.pt --device auto --input-size 512 --overwrite
```

### 6. CVAT 업로드용 zip 생성
```bash
python src/make_cvat_zip.py --dataset-dir 3.0ds --output-zip 3.0ds_for_cvat.zip
```

## 참고

- CVAT용 마스크는 `0=background`, `1=coil`인 **1채널 index mask**다.
- 일반 이미지 뷰어에서는 거의 검정처럼 보일 수 있지만, CVAT 호환성을 위해 의도적으로 이 형식을 사용한다.
