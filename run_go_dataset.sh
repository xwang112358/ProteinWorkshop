#!/usr/bin/env bash
set -euo pipefail

# Export environment variables
export ROOT_DIR="/home/wangx86/ICLR-2026/ProteinWorkshop"
export DATA_PATH="/data/oliver_lab/wangx86/ps_data"

cd "$ROOT_DIR"

# GPU IDs to use (ordered)
GPUS=(0 1)

# Dataset
DATASET="go_proteinshake"
TASK="multilabel_graph_classification"

# Create logs directory
mkdir -p "logs/baselines-go"

# Splits
SPLITS=("random" "structure")

# Models and their features
declare -A MODEL_FEATURES=(
  [gcpnet]="all_equivariant_ca"
  [gear_net]="all_invariant_ca"
  [gear_net_edge]="all_invariant_ca"
)

# Models order
MODELS_ORDER=(gcpnet gear_net gear_net_edge)

# Counter for GPU assignment
gpu_idx=0

# Run experiments for GO dataset on both splits
for split in "${SPLITS[@]}"; do
  dataset_config="go_${split}"

  for model in "${MODELS_ORDER[@]}"; do
    features="${MODEL_FEATURES[$model]}"
    log_file="logs/baselines-go/${dataset_config}_${model}.log"
    gpu="${GPUS[$gpu_idx]}"

    echo "Running: ${model} on ${dataset_config} (GPU ${gpu}) -> ${log_file}"

    CUDA_VISIBLE_DEVICES="${gpu}" python proteinworkshop/train.py \
      "dataset=${dataset_config}" \
      "task=${TASK}" \
      "encoder=${model}" \
      "features=${features}" \
      "test=True" \
      "hydra.run.dir=outputs/train-go/${dataset_config}/${model}" \
      "trainer.max_epochs=60" \
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
echo "All GO dataset experiments completed!"
