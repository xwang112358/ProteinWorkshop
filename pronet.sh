#!/usr/bin/env bash
set -euo pipefail

# GPU IDs to use (ordered) — GPUs 7 and 6
GPUS=(7 6)

# Create logs directory
mkdir -p "logs/pronet"

# Datasets and their task types (lookup)
declare -A DATASET_TASKS=(
  [go_proteinshake]="multilabel_graph_classification"
  [ec_proteinshake]="multiclass_graph_classification"
  [scop_proteinshake]="multiclass_graph_classification"
)

# Deterministic dataset iteration order: ec -> scop -> go
DATASETS_ORDER=(ec_proteinshake scop_proteinshake go_proteinshake)

# Splits (deterministic)
SPLITS=("random" "scaffold")

# ProNet model configuration
MODEL="pronet"
FEATURES="pronet_backbone"

# Counter for GPU assignment
gpu_idx=0

# Run experiments
for dataset in "${DATASETS_ORDER[@]}"; do
  task="${DATASET_TASKS[$dataset]}"
  dataset_short="${dataset/_proteinshake/}"

  for split in "${SPLITS[@]}"; do
    dataset_config="${dataset_short}_${split}"

    # Skip if log already exists (meaning it was run)
    log_file="logs/pronet/${dataset_config}_${MODEL}.log"
    if [[ -f "$log_file" ]]; then
      echo "Skipping: ${MODEL} on ${dataset_config} (log exists)"
      continue
    fi

    gpu="${GPUS[$gpu_idx]}"

    echo "Running: ${MODEL} on ${dataset_config} (GPU ${gpu}) -> ${log_file}"

    CUDA_VISIBLE_DEVICES="${gpu}" python proteinworkshop/train.py \
      "dataset=${dataset_config}" \
      "task=${task}" \
      "encoder=${MODEL}" \
      "features=${FEATURES}" \
      "test=True" \
      "hydra.run.dir=outputs/train-pronet/${dataset_config}/${MODEL}" \
      >"${log_file}" 2>&1 &

    # Move to next GPU
    gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} ))

    # When we've dispatched to all GPUs, wait for one to finish
    if (( gpu_idx == 0 )); then
      wait -n
    fi
  done
done

# Wait for all background jobs to complete
wait
echo "All experiments completed!"
