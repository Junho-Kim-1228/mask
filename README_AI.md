# README_AI

AI 학습, 프리라벨링, masked image 생성에 필요한 핵심 명령어만 빠르게 다시 보기 위한 요약 문서다.

이 문서는 AI 파이프라인 명령어만 빠르게 다시 보기 위한 요약 문서다.  
전체 배경, 시도 기록, 파일 역할, 포트폴리오 설명은 [README.md](/mnt/c/users/wnsgh/kjhdev/mask_project/README.md)를 기준으로 본다.

## 현재 기준

- 메인 모델: `models/coil_unetpp_effb4_scratch_v8_best.pt`
- 추천 추론 크기: `512`
- 추천 프리라벨 threshold: `0.30`
- hard case source: `4.2ds`

## 핵심 명령어

### train/val 데이터 준비

```bash
python src/prepare_cvat_dataset_ai.py --source-dir 1.0ds --source-dir 2.1ds --source-dir 3.2ds --source-dir 4.2ds --output-dir prepared_trainset_v8 --val-ratio 0.2 --hard-val-source 4.2ds --hard-val-count 5 --overwrite
```

### hard-case 재학습

```bash
python src/train_ai.py --train-images-dir prepared_trainset_v8/train/images --train-masks-dir prepared_trainset_v8/train/masks --val-images-dir prepared_trainset_v8/val/images --val-masks-dir prepared_trainset_v8/val/masks --checkpoint-path models/coil_unetpp_effb4_scratch_v8_best.pt --last-checkpoint-path models/coil_unetpp_effb4_scratch_v8_last.pt --input-size 512 --batch-size 1 --epochs 80 --lr 1e-4 --device auto --oversample-source 4.2ds --oversample-factor 16.0 --tversky-alpha 0.20 --tversky-beta 0.80 --focal-gamma 1.5 --boundary-weight 3.0 --focus-val-source 4.2ds --best-metric focus_val_tversky --save-every 20
```

### raw 이미지 프리라벨링

```bash
python src/prelabel_cvat_dataset_ai.py --input-dir dataset --output-dir 4.3ds --model-path models/coil_unetpp_effb4_scratch_v8_best.pt --device auto --input-size 512 --mask-threshold 0.30 --min-component-area 64 --outer-recover-kernel 0 --overwrite
```

### 코일만 남긴 masked image 생성

```bash
python src/apply_ai_mask.py --input-dir dataset --output-dir output/coil_only_ai --model-path models/coil_unetpp_effb4_scratch_v8_best.pt --device auto --input-size 512 --mask-threshold 0.30 --min-component-area 64 --outer-recover-kernel 0 --overwrite
```

### CVAT 업로드용 zip 생성

```bash
python src/make_cvat_zip.py --dataset-dir 4.3ds --output-zip 4.3ds_for_cvat.zip
```

## 참고

- CVAT용 마스크는 `0=background`, `1=coil`인 1채널 index mask다.
- 일반 뷰어로 보면 거의 검정처럼 보일 수 있지만, 형식상 정상이다.
- threshold `0.50`보다 `0.30`에서 경계와 검정 불량 보존이 더 나았다.
