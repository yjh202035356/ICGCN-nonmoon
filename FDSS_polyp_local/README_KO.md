# FDSS-Net 로컬 비교 실험

두 번째 비교 모델: **FDSS-Net**. P2TNet의 공식 저장소에 실행 코드가 없어 대체한다.

- 논문: Wang et al., *FDSS-Net: feature enhancement and dual-stream semantic mixture network for polyp segmentation*
- 출처: Scientific Reports 16, 21509, 2026-05-11
- DOI: https://doi.org/10.1038/s41598-026-49070-5
- 공식 코드: https://github.com/mytest2026-star/FDSS-Net
- 고정 commit: `ebe9ccaee6c96b5c23a311c14dde3f3222bfee52`

이 패키지는 논문의 보고 수치나 저자가 폴립으로 학습한 체크포인트를 가져오지 않는다.
ImageNet으로 사전학습한 PVT-v2 B2 backbone만 초기화에 사용한다.
서버에서 직접 학습한 마지막 epoch 모델로 직접 예측하고 기존 CRSS/SAM2 평가 도구로 측정한다.

## 실행 순서

ZIP을 다운로드하여 서버 `/home/prml/StudentsWork/Jihoon/CRSS-SAM`에 업로드한다.

```bash
cd /home/prml/StudentsWork/Jihoon/CRSS-SAM
ls -lh FDSS_polyp_local.zip &&
python -m zipfile -e FDSS_polyp_local.zip .

conda create -n fdssnet-polyp --clone sam2unet-polyp -y &&
GPU=1 bash FDSS_polyp_local/run.sh setup &&
GPU=1 bash FDSS_polyp_local/run.sh smoke
```

`SMOKE PASS`가 나온 뒤 아래 명령으로 본 실험을 시작한다.
smoke는 첫 16장 배치로 세 가지 입력 크기에서 역전파한다. 결과 가중치는 버린다.
메모리 부족 등의 오류가 나오면 그 로그부터 확인한다. 배치 크기를 자동으로 바꾸지 않는다.

```bash
mkdir -p experiments/polyp1450/seed123/fdss_net/logs
nohup env GPU=1 bash FDSS_polyp_local/run.sh all \
  > experiments/polyp1450/seed123/fdss_net/logs/launcher.log 2>&1 &
tail -f experiments/polyp1450/seed123/fdss_net/logs/launcher.log
```

`all`은 데이터 검사 → 100 epoch 학습 → 테스트 798장 예측 → CPU 평가 순으로 실행한다.
최종 `COMPLETE: FDSS local training, inference, evaluation`이 나오면 완료다.
위 nohup 명령을 썼다면 Ctrl+C는 tail만 종료한다. 같은 실험을 중복 실행하지 않는다.

## 비교 설정

| 항목 | 설정 |
|---|---|
| 학습 | 기존 Polyp1450의 Kvasir 900장 + ClinicDB 550장 |
| 테스트 | Kvasir 100, ClinicDB 62, ColonDB 380, ETIS 196, CVC-300 60 |
| seed | 123 |
| 입력 | 기준 352; 학습 시 256/352/448의 3개 크기 |
| 배치 | 16; 마지막 배치 10, 총 91배치/epoch |
| 학습 길이 | 정확히 100 epoch; 배치당 3회, epoch당 273 optimizer step |
| optimizer | AdamW, lr 1e-4, weight decay 1e-4 |
| LR 감소 | 50, 90 epoch 시작 전에 0.5배 |
| 손실 | 공식 structure_loss 및 5개 출력 가중치 1.2/0.6/0.3/0.15/0.1 유지 |
| 선택 | 테스트를 학습 중 사용하지 않고 100번째 epoch 선택 |
| GPU | CUDA_VISIBLE_DEVICES=1 |
| 환경 | 새 fdssnet-polyp 환경, Torch 2.4.0 cu121 / torchvision 0.19.0 |
| 평가 | 기존 .venv-polyp-eval과 evaluate_polyp_masks.py 재사용 |

서로의 학습 길이·구조·사전학습 데이터까지 동일한 실험은 아니다.
같은 학습/테스트 분할과 같은 평가 구현에서 각 모델을 재학습하는 비교이며, 학습 예산 차이는 보고한다.
100 epoch가 얼마나 걸리는지는 첫 epoch 로그로 추정한다. SAM2-UNet보다 긴 실험이다.

## 공식 코드와의 차이 및 기록

1. `lib/pvt.py` 85행의 BatchNorm 뒤에 쉼표가 빠져 있다. 파일을 덮어쓰지 않고 메모리에 로딩할 때 그 한 곳만 보정한다. 모듈·forward 계산을 추가하거나 제거하지 않는다.
2. 공식 train 함수의 마지막 테스트/체크포인트 선택 블록만 제외한다. batch/scale/augmentation/손실/optimizer 계산은 그대로 실행한다. 실제 적용한 함수를 `adapted_training_function.py`에 기록한다.
3. 공식 `range(1,100)`은 99 epoch를 수행한다. 이 패키지는 명시적으로 100 epoch를 수행한다.
4. 공식 README의 Python 3.12 / Torch 2.8 대신 서버에서 이미 검증한 Python 3.10 / Torch 2.4 cu121을 별도 환경에 복제한다. 환경 차이를 재현성 기록에 명시한다.
5. 공식 손실은 `reduction='mean'`의 scalar BCE를 사용한다. 이를 수정한 픽셀별 가중 BCE로 바꾸지 않는다. 따라서 손실 공식의 엄밀한 논문 재현보다는 공개 구현 기반의 통제된 재학습 결과로 표기한다.

검사 JSON, source commit, 초기 backbone SHA256, 학습 설정, 설치 패키지 목록, 최종 체크포인트 SHA256을 저장한다.
PVT ImageNet 가중치 출처: https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth
SHA256: `80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d`

## 결과 위치

기준 폴더: `experiments/polyp1450/seed123/fdss_net/`

- 최종 모델: `checkpoints/FDSS-Net-100.pth`
- 마지막 완료 epoch 모델: `checkpoints/latest.pth`
- 재시작용 모델/optimizer/RNG: `checkpoints/resume.pth`
- 현재 완료 epoch: `progress.json`
- 재현성 기록: `training_started.json`, `training_completed.json`, `pip_freeze.txt`
- 테스트 예측: `predictions/<dataset>/*.png`
- 원래 sigmoid 확률: `predictions/raw_probabilities/<dataset>/*.npz`
- 평가: `metrics/FDSS_Net_local.json`, `.csv`

NPZ의 `semantic` 배열 이름은 기존 평가 도구와의 호환을 위한 이름이다. FDSS-Net의 최종 출력 하나를 뜻한다.
평가는 threshold_mean_dice/IoU와 저장 맵의 0.5 지표, 원래 확률의 0.5 지표를 별도로 기록한다.

## 중단 후 이어서 실행

프로세스가 중단된 것을 먼저 확인하고 실행한다. epoch 중간은 저장하지 않으며 마지막으로 **완료한** epoch부터 복원한다.

```bash
nohup env GPU=1 bash FDSS_polyp_local/run.sh resume \
  >> experiments/polyp1450/seed123/fdss_net/logs/launcher.log 2>&1 &
tail -f experiments/polyp1450/seed123/fdss_net/logs/launcher.log
```

resume는 학습만 수행한다. `TRAIN COMPLETE` 후:

```bash
GPU=1 bash FDSS_polyp_local/run.sh export &&
bash FDSS_polyp_local/run.sh evaluate
```

학습이 이미 완료됐다면 재학습하지 않고 export/evaluate만 실행한다.
이미 생성한 예측을 덮어쓰지 않는다. 실패한 부분 예측이 남았다면 별도 이름으로 보관한 뒤 다시 export한다.
`WORK_ROOT`를 지정하면 다른 폴더로 작업할 수 있다. 기존 결과를 지우거나 수정하지 않는다.

## 검증 범위

`validation.json`은 패키지 제작 환경의 실제 CPU 검사 결과다.
PVT 초기화, 역전파, 세 production 입력 크기의 출력, 테스트 사용 차단, 저장/복원 및 pairing을 검사했다.
서버 GPU의 메모리 적합성과 전체 학습 완료는 서버 smoke 및 최종 로그로 확인해야 한다.
