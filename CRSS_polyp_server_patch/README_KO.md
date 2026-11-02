# CRSS-SAM Polyp1450 서버 실행 안내

저장소 `https://github.com/yjh202035356/ICGCN-nonmoon.git`의 `b8fdd99`에 추가된 teacher 생성 코드를 확인하고 준비한 실험 연결 패치입니다. 원격 GitHub에는 push하지 않았습니다. 서버에서 아래 패치를 적용하면 실행 스크립트가 추가됩니다.

현재 완료한 것은 코드 연결 및 CPU 기능 검사입니다. 실제 BiomedCLIP 체크포인트를 이용한 1,450장 teacher 생성과 TITAN RTX 학습/추론은 아직 실행하지 않았습니다. 동봉한 검사 결과는 작은 합성 데이터와 작은 테스트 모델로 얻은 기능 검사이며, 폴립 성능 결과가 아닙니다.

## 1. 패치 적용

`CRSS_polyp_server_patch.zip`을 서버의 `/home/prml/StudentsWork/Jihoon/CRSS-SAM/`에 올린 다음:

```bash
cd /home/prml/StudentsWork/Jihoon/CRSS-SAM
conda activate simplevilmedsam
python -m zipfile -e CRSS_polyp_server_patch.zip .
bash CRSS_polyp_server_patch/apply_patch.sh .
```

적용 스크립트는 먼저 `git apply --check`를 실행합니다. 이미 같은 패치가 적용되어 있으면 다시 적용하지 않습니다. 다른 변경과 충돌하면 자동으로 덮어쓰지 않습니다. `git diff`로 확인할 수 있습니다. 이 과정은 코드 및 설정만 추가/수정하며 데이터나 체크포인트를 압축에 포함하지 않습니다.

## 2. 데이터 준비와 teacher 3장 검사

기본 데이터 경로:

- 원본 학습: `dataset/TrainDataset/images` 또는 `dataset/TrainDataset/image`, `masks`
- CRSS 학습: `data/Polyp1450/train/images`, `labels`, `attribution_map`
- 테스트: `dataset/TestDataset/{Kvasir,CVC-ClinicDB,CVC-ColonDB,ETIS-LaribPolypDB,CVC-300}/images`, `masks`

학습 원본은 Kvasir 900장 + ClinicDB 550장입니다. 랜덤 재분할하지 않습니다. 준비 단계는 원본 images/masks 폴더를 CRSS 위치에 심볼릭 링크로 연결하며, 이미 해당 위치에 파일이 있으면 원본과 동일한지 검사합니다.

```bash
GPU=0 bash scripts/run_polyp1450.sh prepare
ls -lh checkpoints/BiomedCLIP-vit-bert-hf/config.json
GPU=0 bash scripts/run_polyp1450.sh teacher-smoke
```

teacher smoke는 전체 학습 캐시와 다른 `experiments/polyp1450/seed123/teacher_smoke`에 3장만 생성합니다. 로그는 `experiments/polyp1450/seed123/logs/teacher_smoke.log`입니다.

기본 로컬 BiomedCLIP 경로는 기존 Simple-ViLMedSAM에서 사용한 `checkpoints/BiomedCLIP-vit-bert-hf`입니다. CRSS의 checkpoints 링크가 그 폴더를 노출한다는 전제입니다. 경로가 다르면 다음처럼 실제 경로를 지정합니다:

```bash
GPU=0 BIOMEDCLIP_MODEL=/실제/BiomedCLIP-vit-bert-hf/경로 bash scripts/run_polyp1450.sh teacher-smoke
```

이번 teacher 설정은 text=`polyp`, vbeta=0.1, vvar=0.1, vlayer=8, seed=123입니다. 초기 map은 추가한 원본 IBA 함수를 그대로 호출하며, 하이퍼파라미터 탐색은 하지 않습니다. CLIP 가중치를 고정하고 IBA의 bottleneck만 최적화합니다. 원본 크기에 복원해 grayscale PNG로 저장합니다. 오버레이는 학습에 사용하지 않습니다.

이미 전체 teacher를 생성했다면 새 생성을 실행하기 전에 다음 audit로 확인할 수 있습니다. 기존 생성 설정/로그는 보관하세요. 기존 파일의 개수·크기 검사는 teacher의 생성 설정까지 증명하지 않습니다.

```bash
GPU=0 bash scripts/run_polyp1450.sh audit
```

## 3. teacher 1,450장 생성

3장 검사가 성공했고 전체 teacher 캐시를 아직 생성하지 않았다면:

```bash
GPU=0 bash scripts/run_polyp1450.sh teacher
```

새 생성기는 PNG와 `teacher_generation.json`을 저장합니다. 같은 설정·입력·모델의 정상 캐시는 재실행 시 재사용합니다. 설정이나 모델 파일이 달라지면 재사용을 차단합니다. 이미지 stem별 난수 시드를 기록하므로 중단 후 재개해도 새로 생성되는 맵의 난수 설정은 같습니다. 기존 원본 생성기가 만든 캐시에는 이 provenance 파일이 없으므로 새 생성기로 자동 덮어쓰지 않습니다. 해당 캐시를 사용하려면 먼저 audit를 실행하고 기존 생성 로그를 보관하세요.

생성 완료 뒤 모든 image/label/map의 stem과 원본 해상도가 일치하는지 검사합니다. teacher가 빠진 이미지를 조용히 제외하지 않습니다. constant/빈 출력, 잘못된 해상도, 중복 stem은 오류로 처리합니다.

## 4. GPU 학습 smoke

```bash
GPU=0 bash scripts/run_polyp1450.sh smoke
```

전체 1,450개 triplet의 준비 여부를 먼저 검사한 뒤, LoRA+heads와 resolver를 각각 2 batch, 1 epoch만 실행합니다. smoke 출력은 `smoke_heads`, `smoke_resolver`에 저장됩니다. 실험 결과 표에는 사용하지 않습니다.

확인할 로그:

- `experiments/polyp1450/seed123/logs/heads_smoke.log`
- `experiments/polyp1450/seed123/logs/resolver_smoke.log`

smoke 결과와 GPU 메모리를 확인한 뒤 전체 학습을 실행합니다. 서버 conda 환경의 PyTorch 2.1.0+cu121을 교체하도록 지시하지 않습니다.

## 5. 전체 학습

```bash
GPU=0 bash scripts/run_polyp1450.sh train
```

터미널 종료 후에도 유지하려면:

```bash
mkdir -p experiments/polyp1450/seed123/logs
nohup env GPU=0 bash scripts/run_polyp1450.sh train > experiments/polyp1450/seed123/logs/launcher.log 2>&1 &
tail -f experiments/polyp1450/seed123/logs/launcher.log
```

이번 연결은 **SAM ViT-H + late Q/V LoRA + dual heads + teacher refinement v2 + resolver v2**입니다. 새로운 SAM 모델로 초기화하고 trainable 모듈도 새로 초기화합니다. 예전 Kvasir 모델을 warm-start하지 않습니다.

| 항목 | 설정 |
|---|---|
| SAM checkpoint | `/home/prml/StudentsWork/Jihoon/CRSS-SAM/checkpoints/sam_vit_h_4b8939.pth` |
| mid block / LoRA blocks | 15 / 16–31 |
| LoRA rank / alpha | 4 / 4 |
| LoRA+heads | 60 epochs, lr=3e-4 |
| Teacher refinement | temperature=0.1, teacher_weight=0.5 |
| Resolver v2 | 40 epochs, lr=1e-3, tau=0.1 |
| Batch / accumulation | 1 / 2 |
| SAM input / loss size | 1024 / 256 |
| Seed | 123 |
| Dataset | 전체 1,450장, max_train_batches=0 |
| Checkpoint selection | 사전에 고정한 마지막 epoch |

학습 중 공식 테스트나 validation을 호출하지 않습니다. `--fixed_schedule`과 `--val_root`를 동시에 주면 오류로 처리합니다. 단계별 마지막 파일은 `heads60/latest.pth`, `resolver40/latest.pth`입니다. 이 모드에서는 validation 기반 `best.pth`를 새로 만들지 않습니다.

teacher refinement는 LoRA 학습 중 현재 deep feature로 teacher target을 보정합니다. 원본 `train_crss_teacher_refine_v2.py`는 frozen SAM heads-only 학습이므로, 그 파일을 LoRA 학습인 것처럼 호출하지 않습니다. 새 옵션 `--teacher_refine_v2`가 LoRA 학습에 보정을 연결합니다.

기존 resolver v2는 LoRA를 복원하지 않았습니다. 수정된 resolver는 heads checkpoint의 LoRA 설정과 전체 LoRA tensor를 검증·복원한 뒤 SAM/LoRA/heads를 고정하고 resolver만 학습합니다. resolver 파일에는 사용한 heads checkpoint의 SHA256이 기록됩니다.

## 6. 5개 테스트셋 추론

아직 서버에 TestDataset이 없다면 앞서 제공한 `TestDataset.zip`을 `dataset` 아래에 풀어 위 구조로 준비합니다. zip에 `TestDataset` 최상위 폴더가 있는지 확인하세요. 이미지와 라벨 수는 Kvasir 100, ClinicDB 62, ColonDB 380, ETIS 196, CVC-300 60, 합계 798장입니다.

```bash
GPU=0 bash scripts/run_polyp1450.sh export
```

추론 전에 읽을 수 있는 image/mask와 0개 train/test decoded-RGB 일치를 검사합니다. 이 검사는 정확히 같은 이미지의 중복을 검사하며 환자/동영상 독립성까지 증명하지는 않습니다. 공식 테스트 간 중복 이미지는 임의 삭제하지 않습니다.

동일한 checkpoint에서 semantic, structural, naive_average, resolver_v2를 추론합니다. 이 4개 출력 비교는 별도로 재학습한 모듈 제거 실험과 구분해야 합니다.

CLIP이나 teacher map 없이 추론합니다. 정답 mask는 pairing/원본 해상도 검사에만 사용하며 모델 입력에는 전달하지 않습니다. native logits을 1024로 올리고, bottom/right padding을 제거한 뒤 원본 해상도로 복원합니다. 결합 확률은 logit으로 변환한 다음 동일한 복원 경로를 적용합니다. 따라서 naive average는 native grid에서의 평균입니다.

저장 위치:

- `predictions/{semantic,structural,naive_average,resolver_v2}/{dataset}/{stem}.png`: 논문 비교용 per-image min-max 정규화 PNG
- `predictions/raw_probabilities/{dataset}/{stem}.npz`: 정규화 전 float32 확률 4개
- `predictions/export_manifest.json`: checkpoint hash, epoch, count, 복원 규칙

완료되지 않은 학습이나 smoke checkpoint는 전체 benchmark export에서 거부합니다. smoke export는 `--max-images`를 지정해야 합니다. 출력을 섞지 않도록 export는 새 출력 폴더를 요구합니다.

## 7. 점수 계산

학습 환경의 torch/numpy를 건드리지 않도록, 평가에 필요한 패키지는 Python 3.10의 별도 venv에 설치합니다. 앞서 제공한 Python 3.12용 numpy 2.3.4 pin을 `simplevilmedsam`에 설치하지 마세요.

```bash
python -m venv .venv-polyp-eval
.venv-polyp-eval/bin/python -m pip install -r requirements_polyp_eval_python310.txt
PATH="$PWD/.venv-polyp-eval/bin:$PATH" bash scripts/run_polyp1450.sh evaluate
```

`metrics/{method}.csv`, `.json`이 생성됩니다. 논문 비교는 **256개 threshold의 평균 Dice/IoU**를 사용합니다. 정규화 PNG의 threshold=0.5와 실제 raw probability의 threshold=0.5는 따로 기록됩니다. 이 세 가지 수치를 같은 표의 같은 열로 섞지 않습니다.

비교 방법은 SAM2-UNet(2026 정식 출판; 최초 preprint는 2024), P2TNet(2026) 두 개입니다. SAM2-UNet의 저자 공개 예측 맵은 앞선 CPU 평가에서 논문 표의 10개 값을 소수 셋째 자리까지 확인했습니다. P2TNet은 공개 실행 코드가 없어 논문 보고값 비교입니다. P2TNet의 파일 단위 split 일치까지 확인했다는 의미는 아닙니다.

| Test dataset | SAM2-UNet Dice / IoU | P2TNet Dice / IoU |
|---|---|---|
| Kvasir | 0.928 / 0.879 | 0.916 / 0.865 |
| CVC-ClinicDB | 0.907 / 0.856 | 0.930 / 0.878 |
| CVC-ColonDB | 0.808 / 0.730 | 0.794 / 0.713 |
| ETIS-LaribPolypDB | 0.796 / 0.723 | 0.768 / 0.682 |
| CVC-300 | 0.894 / 0.827 | 0.875 / 0.802 |

원문: https://doi.org/10.1007/s44267-025-00106-w ; https://doi.org/10.1049/ipr2.70366

CRSS 수치는 실제 GPU 학습/추론 및 평가가 끝난 뒤에 채웁니다. 과거 Kvasir val/test 결과를 이 새 5개 데이터셋 비교 표에 넣지 않습니다.

전체 학습 두 단계를 따로 실행하려면 `train-heads`, `train-resolver`를 사용할 수 있습니다. heads60이 완료된 뒤 resolver 단계에서 오류가 발생했다면 `train-resolver`만 재실행하세요. 이 스크립트는 중간 epoch optimizer 상태를 복원하는 학습 resume를 제공하지 않습니다. teacher 생성의 캐시 재개와 학습 resume는 구분합니다.
