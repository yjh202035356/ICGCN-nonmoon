# SAM2-UNet 서버 직접 재현

CRSS의 완료된 실험은 다시 실행하지 않습니다. 이 파일은 공식 SAM2-UNet 코드로 서버에서 학습한 모델의 결과만 평가합니다.

## 실행 순서

아래 ZIP을 PC에서 내려받아 Remote SSH로 `/home/prml/StudentsWork/Jihoon/CRSS-SAM/SAM2_polyp_local.zip`에 업로드합니다.

```bash
cd /home/prml/StudentsWork/Jihoon/CRSS-SAM
python -m zipfile -e SAM2_polyp_local.zip .
conda create -n sam2unet-polyp python=3.10 -y
GPU=1 bash SAM2_polyp_local/run.sh setup
```

환경이 이미 있다면 conda create만 생략합니다. 기존 simplevilmedsam 환경에서 setup을 실행해도 패키지는 새 conda 환경에 설치됩니다. 원본 SAM 체크포인트와 다른 SAM2 Hiera-L 사전 학습 체크포인트를 다운로드합니다. 폴립 학습 완료 가중치를 다운로드하지 않습니다.

```bash
mkdir -p experiments/polyp1450/seed123/sam2_unet/logs
nohup env GPU=1 bash SAM2_polyp_local/run.sh all \
  > experiments/polyp1450/seed123/sam2_unet/logs/launcher.log 2>&1 &
tail -f experiments/polyp1450/seed123/sam2_unet/logs/launcher.log
```

all은 데이터 검사 → 20에포크 학습 → 테스트 798장 직접 추론 → 기존 평가 venv로 평가를 수행합니다. GPU 0을 사용하지 않습니다. 결과는 `experiments/polyp1450/seed123/sam2_unet/metrics/SAM2_UNet_local.json`과 CSV에 저장합니다. 추론 실패 시 학습을 반복하지 않고 export와 evaluate만 순서대로 실행할 수 있습니다. 기존 예측을 덮어쓰지 않습니다. export 중단 후 일부 예측이 생겼으면 자동 재개는 지원하지 않으므로 로그를 확인하여 해당 폴더를 먼저 별도 보관해야 합니다.

## 실험 기준

- 공식 저장소 https://github.com/WZH0120/SAM2-UNet
- 고정 커밋 01598e5e9912ffb23f965ecbebf4d1dfecbaa56e
- 동일 학습 1450장과 테스트 798장. 기존 teacher는 SAM2-UNet에서 사용하지 않습니다.
- 공식 기본값: Hiera-L, 입력 352×352, 배치 12, 20에포크, AdamW, lr 0.001, weight decay 0.0005, cosine schedule, 수직·수평 반전, 3출력 deep supervision.
- 시드 123. 테스트로 모델을 선택하지 않고 마지막 20에포크 체크포인트만 평가합니다.
- 원본 train.py, dataset.py, 모델 코드와 학습 손실을 수정하지 않습니다. 원본 train.py의 `reduce='none'` 구식 호출도 그대로 유지합니다. 이 구식 호출은 PyTorch에서 요소별 BCE를 요청하는 `reduction='none'`과 다르게 동작하므로, 이번 실행을 수정 없는 **저자 공개 구현 재현**으로 기록해야 하며 논문 수식을 정확히 구현했다고 주장하지 않습니다. 이 항목은 training_started/completed.json에도 기록합니다.
- 공식 test.py와 같은 352 입력 정규화, 최종 출력 선택, 원본 GT 크기 bilinear 복원, sigmoid, 이미지별 minmax PNG를 사용합니다. 마스크/이미지는 stem으로 엄격히 대응시킵니다. 원시 sigmoid 확률도 저장하여 임계값 평균, 저장 맵의 0.5, 원시 확률의 0.5를 각각 구분합니다.
- 원시 확률 NPZ의 semantic 키는 기존 평가 코드의 단일 출력 호환을 위한 배열명입니다. CRSS의 의미 헤드를 실행한다는 뜻은 아닙니다.
- 별도 torch 2.4.0+cu121 환경은 공식 요구사항에 맞춥니다. CRSS의 기존 torch 2.1.0 환경과 패키지는 수정하지 않습니다. 같은 GPU·데이터·평가 기준이며, 모델별 공식 입력 크기와 학습 설정, torch 버전 차이는 보고서에 명시합니다.
- 아직 실제 TITAN RTX에서 실행하지 않았습니다. CUDA 동작과 메모리 사용은 setup/서버 학습 로그로 확인합니다. 메모리 오류가 나면 배치를 임의로 줄여 재실행하지 않고 로그를 공유해 설정 변경을 기록합니다.

## 두 번째 방법

2026-10-12 확인한 https://github.com/zhangxiaolijlu/P2Net 는 README만 있으며 실행 코드가 없습니다. P2TNet 학습은 이 ZIP에 포함되지 않습니다. 저자 구현 확보 또는 다른 실행 가능한 2026 비교 방법 선정이 필요합니다.
