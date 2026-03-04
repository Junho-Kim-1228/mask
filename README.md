# mask_project

코일 유사 이미지를 대상으로 전처리 및 마스킹을 수행하는 파이프라인입니다.

## 변경 사항

`src/process_all.py`는 이제 **자동 누끼 방식(rembg) 우선 + adaptive 보조** 파이프라인으로 동작합니다.

- 고정 타입 매핑(`-1/-2/-3`)과 정적 A/B/C 마스크 참조를 사용하지 않습니다.
- 기본 분할기는 `rembg`이며, 갤럭시 자동 누끼처럼 전경(코일)을 자동 분리합니다.
- `rembg` 결과가 비정상(너무 작거나/큰 영역)일 때는 adaptive 방식으로 자동 보정할 수 있습니다.
- adaptive 보정 시에는 여러 후보(HSV/LAB/조합)를 생성하고 품질 점수로 최적 후보를 선택합니다.
- 필요하면 `GrabCut` 정밀화를 추가로 적용해 경계 품질을 개선합니다.
- 최종 단계에서 가장자리 `trim`을 적용해 배경 잔여를 더 깎아냅니다(기본: `trim-level=3`).
  - rembg 사용 시: alpha 신뢰도 기반 선택적 트리밍(안 깎여야 할 본체 보존 강화)
  - adaptive 사용 시: 형태학 기반 트리밍
- 최종 코일 마스크의 내부 구멍은 기본적으로 채워서(홀 제거) 저장합니다.
- 계산된 마스크는 내부에서만 사용하고, 파일로는 저장하지 않습니다.

## 디렉터리 구조

- 입력 이미지: `data/`
- 마스킹 결과 이미지(기본): `output/coil_only/`

## 실행 방법

```bash
python src/process_all.py
```

자주 쓰는 옵션:

```bash
# 앞에서 20장만 빠르게 테스트
python src/process_all.py --max-images 20

# 결과 파일이 이미 있어도 다시 처리
python src/process_all.py --no-skip-existing

# 속도 우선(GrabCut 정밀화 비활성화)
python src/process_all.py --no-grabcut

# adaptive 방식만 강제 사용(rembg 미사용)
python src/process_all.py --segmenter adaptive

# rembg 모델 지정
python src/process_all.py --segmenter rembg --rembg-model isnet-general-use

# rembg 모델 자동 앙상블(auto: isnet-general-use,u2net,u2netp)
python src/process_all.py --segmenter rembg --rembg-model auto

# rembg 실패 시 adaptive 자동 보정 끄기
python src/process_all.py --no-fallback-adaptive

# rembg 점수가 낮을 때 adaptive 보정 기준 조정(기본 0.2)
python src/process_all.py --fallback-score-threshold 1.2

# 더 강하게 코일만 남기기
python src/process_all.py --trim-level 4

# 최대 강도로 깎기
python src/process_all.py --trim-level 5

# 초강력으로 더 깎기
python src/process_all.py --trim-level 8

# 참고: trim-level 6~8은 얇은 돌출/가지를 우선 제거하는 강한 프루닝 모드입니다.

# trim 끄기
python src/process_all.py --trim-level 0

# 내부 구멍 채우기 끄기(기본은 켜짐)
python src/process_all.py --no-fill-holes

# 특정 파일명만 처리
python src/process_all.py --name-contains "251010_084003_A35W_6-2 [16]"

# 입출력 경로 직접 지정
python src/process_all.py \
  --input-dir data \
  --output-masked-dir output/my_masked
```

## 의존성 설치

아래 명령으로 패키지를 설치하세요.

```bash
pip install -r requirements.txt
```

`rembg` 사용 시 최초 실행에서 모델 다운로드가 발생할 수 있습니다.
