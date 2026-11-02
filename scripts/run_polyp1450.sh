#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
train_root="${POLYP_TRAIN_ROOT:-data/Polyp1450/train}"
test_root="${POLYP_TEST_ROOT:-dataset/TestDataset}"
source_root="${POLYP_SOURCE_ROOT:-dataset/TrainDataset}"
sam_checkpoint="${SAM_CHECKPOINT:-/home/prml/StudentsWork/Jihoon/CRSS-SAM/checkpoints/sam_vit_h_4b8939.pth}"
teacher_model="${BIOMEDCLIP_MODEL:-checkpoints/BiomedCLIP-vit-bert-hf}"
work="${POLYP_WORK_ROOT:-experiments/polyp1450/seed123}"
stage="${1:-}"
mkdir -p "$work/logs"
audit_teacher() {
    python -m scripts.prepare_polyp1450 --audit-only --train-root "$train_root" --require-teacher --output "$work/data_audit.json"
}
train_heads() {
    local directory="$1" epochs="$2" max_batches="$3"
    python -u train_crss_lora.py --train_root "$train_root" --sam_checkpoint "$sam_checkpoint" \
      --work_dir "$directory" --fixed_schedule --expected_train_count 1450 \
      --epochs "$epochs" --batch_size 1 --accumulation_steps 2 --lr 3e-4 \
      --mid_block 15 --lora_start_block 16 --lora_rank 4 --lora_alpha 4 \
      --teacher_refine_v2 --affinity_temperature 0.1 --teacher_weight 0.5 \
      --seed 123 --num_workers 4 --max_train_batches "$max_batches"
}
train_resolver() {
    local directory="$1" backbone="$2" epochs="$3" max_batches="$4"
    python -u train_resolver_v2.py --train_root "$train_root" --sam_checkpoint "$sam_checkpoint" \
      --heads_checkpoint "$backbone" --work_dir "$directory" --fixed_schedule \
      --expected_train_count 1450 --epochs "$epochs" --batch_size 1 --accumulation_steps 2 \
      --lr 1e-3 --seed 123 --num_workers 4 --max_train_batches "$max_batches"
}
case "$stage" in
  prepare)
    python -m scripts.prepare_polyp1450 --source "$source_root" --train-root "$train_root" --output "$work/data_audit.json"
    ;;
  teacher-smoke)
    python -u -m scripts.generate_polyp_teacher --input-path "$train_root/images" \
      --output-path "$work/teacher_smoke" --model-path "$teacher_model" --max-images 3 \
      2>&1 | tee "$work/logs/teacher_smoke.log"
    ;;
  teacher)
    python -u -m scripts.generate_polyp_teacher --input-path "$train_root/images" \
      --output-path "$train_root/attribution_map" --model-path "$teacher_model" \
      2>&1 | tee "$work/logs/teacher_full.log"
    audit_teacher
    ;;
  audit)
    audit_teacher
    ;;
  smoke)
    audit_teacher
    train_heads "$work/smoke_heads" 1 2 2>&1 | tee "$work/logs/heads_smoke.log"
    train_resolver "$work/smoke_resolver" "$work/smoke_heads/latest.pth" 1 2 2>&1 | tee "$work/logs/resolver_smoke.log"
    ;;
  train)
    audit_teacher
    train_heads "$work/heads60" 60 0 2>&1 | tee "$work/logs/heads60.log"
    train_resolver "$work/resolver40" "$work/heads60/latest.pth" 40 0 2>&1 | tee "$work/logs/resolver40.log"
    ;;
  train-heads)
    audit_teacher
    train_heads "$work/heads60" 60 0 2>&1 | tee "$work/logs/heads60.log"
    ;;
  train-resolver)
    train_resolver "$work/resolver40" "$work/heads60/latest.pth" 40 0 2>&1 | tee "$work/logs/resolver40.log"
    ;;
  export)
    python -m scripts.prepare_polyp1450 --audit-only --train-root "$train_root" --test-root "$test_root" --output "$work/train_test_audit.json"
    python -u export_polyp_benchmark.py --test-root "$test_root" --sam_checkpoint "$sam_checkpoint" \
      --heads_checkpoint "$work/heads60/latest.pth" --resolver_checkpoint "$work/resolver40/latest.pth" \
      --output "$work/predictions" 2>&1 | tee "$work/logs/export.log"
    ;;
  evaluate)
    for method in semantic structural naive_average resolver_v2; do
      python evaluate_polyp_masks.py --pred-root "$work/predictions/$method" --gt-root "$test_root" \
        --method "CRSS_$method" --provenance local_inference --output "$work/metrics/$method" \
        --raw-root "$work/predictions/raw_probabilities" --raw-key "$method"
    done
    ;;
  *)
    echo "Usage: GPU=0 bash scripts/run_polyp1450.sh {prepare|teacher-smoke|teacher|audit|smoke|train|train-heads|train-resolver|export|evaluate}" >&2
    exit 2
    ;;
esac
