#!/usr/bin/env bash
set -euo pipefail
CRSS_ROOT="${CRSS_ROOT:-/home/prml/StudentsWork/Jihoon/CRSS-SAM}"
CRSS_ROOT="$(cd "$CRSS_ROOT" && pwd)"
KIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$CRSS_ROOT/baselines/FDSS-Net"
WORK_ROOT="${WORK_ROOT:-$CRSS_ROOT/experiments/polyp1450/seed123/fdss_net}"
TEST_ROOT="${POLYP_TEST_ROOT:-$CRSS_ROOT/data/Polyp1450/dataset/TestDataset}"
TRAIN_ROOT="$CRSS_ROOT/data/Polyp1450/train"
ENV_NAME="${FDSS_ENV:-fdssnet-polyp}"
PRETRAINED="$CRSS_ROOT/baselines/pretrained/pvt_v2_b2.pth"
COMMIT=ebe9ccaee6c96b5c23a311c14dde3f3222bfee52
PVT_SHA=80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export PYTHONHASHSEED=123
export PYTHONDONTWRITEBYTECODE=1
cd "$CRSS_ROOT"
mkdir -p "$WORK_ROOT/logs"
run_python() {
  conda run --no-capture-output -n "$ENV_NAME" python -u "$KIT_ROOT/run_baseline.py" "$1" \
    --repo "$REPO_ROOT" --train-root "$TRAIN_ROOT" --test-root "$TEST_ROOT" \
    --pretrained "$PRETRAINED" --work "$WORK_ROOT"
}
audit() {
  conda run --no-capture-output -n "$ENV_NAME" python -u -m scripts.prepare_polyp1450 \
    --audit-only --train-root "$TRAIN_ROOT" --test-root "$TEST_ROOT" \
    --output "$WORK_ROOT/train_test_audit.json"
}
evaluate() {
  "$CRSS_ROOT/.venv-polyp-eval/bin/python" "$CRSS_ROOT/evaluate_polyp_masks.py" \
    --pred-root "$WORK_ROOT/predictions" --gt-root "$TEST_ROOT" --method FDSS_Net_local \
    --provenance local_inference --output "$WORK_ROOT/metrics/FDSS_Net_local" \
    --raw-root "$WORK_ROOT/predictions/raw_probabilities" --raw-key semantic
}
case "${1:-}" in
  setup)
    mkdir -p "$CRSS_ROOT/baselines/pretrained"
    if [[ ! -d "$REPO_ROOT" ]]; then
      git clone https://github.com/mytest2026-star/FDSS-Net.git "$REPO_ROOT"
      git -C "$REPO_ROOT" checkout --detach "$COMMIT"
    fi
    [[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$COMMIT" ]] || { echo 'Unexpected FDSS source revision'; exit 1; }
    conda run --no-capture-output -n "$ENV_NAME" python -m pip install -r "$KIT_ROOT/requirements.txt"
    if [[ ! -s "$PRETRAINED" ]]; then
      curl -fL --retry 3 -o "$PRETRAINED.part" https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth
      mv "$PRETRAINED.part" "$PRETRAINED"
    fi
    [[ "$(sha256sum "$PRETRAINED" | cut -d' ' -f1)" == "$PVT_SHA" ]] || { echo 'PVT checkpoint SHA256 mismatch'; exit 1; }
    conda run --no-capture-output -n "$ENV_NAME" python -c 'import torch, torchvision; assert torch.__version__.split("+")[0] == "2.4.0"; assert torchvision.__version__.split("+")[0] == "0.19.0"; assert torch.cuda.is_available(); print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU:", torch.cuda.get_device_name(0))'
    echo 'SETUP COMPLETE. Run GPU=1 bash FDSS_polyp_local/run.sh smoke'
    ;;
  smoke) audit; run_python smoke ;;
  train) audit; run_python train ;;
  resume) audit; run_python resume ;;
  export) audit; run_python export ;;
  evaluate) evaluate ;;
  all)
    audit
    run_python train
    run_python export
    evaluate
    echo 'COMPLETE: FDSS local training, inference, evaluation'
    ;;
  *) echo 'Usage: GPU=1 bash FDSS_polyp_local/run.sh {setup|smoke|all|train|resume|export|evaluate}'; exit 2 ;;
esac
