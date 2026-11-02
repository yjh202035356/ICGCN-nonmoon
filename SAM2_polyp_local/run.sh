#!/usr/bin/env bash
set -euo pipefail
CRSS_ROOT="${CRSS_ROOT:-/home/prml/StudentsWork/Jihoon/CRSS-SAM}"
CRSS_ROOT="$(cd "$CRSS_ROOT" && pwd)"
KIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$CRSS_ROOT/baselines/SAM2-UNet"
WORK_ROOT="${WORK_ROOT:-$CRSS_ROOT/experiments/polyp1450/seed123/sam2_unet}"
TEST_ROOT="${POLYP_TEST_ROOT:-$CRSS_ROOT/data/Polyp1450/dataset/TestDataset}"
TRAIN_ROOT="$CRSS_ROOT/data/Polyp1450/train"
PRETRAINED="$CRSS_ROOT/baselines/pretrained/sam2_hiera_large.pt"
ENV_NAME="${SAM2_ENV:-sam2unet-polyp}"
COMMIT=01598e5e9912ffb23f965ecbebf4d1dfecbaa56e
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export PYTHONHASHSEED=123
export PYTHONDONTWRITEBYTECODE=1
cd "$CRSS_ROOT"
mkdir -p "$WORK_ROOT/logs"
run_python() {
    conda run --no-capture-output -n "$ENV_NAME" python -u "$KIT_ROOT/run_baseline.py" "$1" \
      --crss-root "$CRSS_ROOT" --repo "$REPO_ROOT" --train-root "$TRAIN_ROOT" \
      --test-root "$TEST_ROOT" --pretrained "$PRETRAINED" --work "$WORK_ROOT"
}
evaluate() {
    "$CRSS_ROOT/.venv-polyp-eval/bin/python" "$CRSS_ROOT/evaluate_polyp_masks.py" \
      --pred-root "$WORK_ROOT/predictions" --gt-root "$TEST_ROOT" --method SAM2_UNet_local \
      --provenance local_inference --output "$WORK_ROOT/metrics/SAM2_UNet_local" \
      --raw-root "$WORK_ROOT/predictions/raw_probabilities" --raw-key semantic
}
case "${1:-}" in
 setup)
    mkdir -p "$CRSS_ROOT/baselines/pretrained"
    if [[ ! -d "$REPO_ROOT" ]]; then
      git clone https://github.com/WZH0120/SAM2-UNet.git "$REPO_ROOT"
      git -C "$REPO_ROOT" checkout --detach "$COMMIT"
    fi
    [[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$COMMIT" ]] || { echo 'Unexpected baseline source revision'; exit 1; }
    conda run -n "$ENV_NAME" python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
    conda run -n "$ENV_NAME" python -m pip install numpy==1.26.4 Pillow==10.4.0 hydra-core==1.3.2 tqdm==4.66.5 imageio==2.34.2
    if [[ ! -s "$PRETRAINED" ]]; then
      curl -fL --retry 3 -o "$PRETRAINED.part" https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
      mv "$PRETRAINED.part" "$PRETRAINED"
    fi
    conda run --no-capture-output -n "$ENV_NAME" python -c 'import torch, torchvision, hydra; assert torch.cuda.is_available(); print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0))'
    echo 'SETUP COMPLETE. Run GPU=1 bash SAM2_polyp_local/run.sh all'
    ;;
 train) run_python train ;;
 export)
    python -m scripts.prepare_polyp1450 --audit-only --train-root "$TRAIN_ROOT" --test-root "$TEST_ROOT" --output "$WORK_ROOT/train_test_audit.json"
    run_python export ;;
 evaluate) evaluate ;;
 all)
    python -m scripts.prepare_polyp1450 --audit-only --train-root "$TRAIN_ROOT" --test-root "$TEST_ROOT" --output "$WORK_ROOT/train_test_audit.json"
    run_python train
    run_python export
    evaluate
    echo 'COMPLETE: local training, inference, evaluation'
    ;;
 *) echo 'Usage: GPU=1 bash SAM2_polyp_local/run.sh {setup|all|train|export|evaluate}'; exit 2 ;;
esac
