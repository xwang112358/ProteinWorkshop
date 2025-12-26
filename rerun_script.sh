#!/usr/bin/env bash
set -euo pipefail

# Export environment variables
export ROOT_DIR="/home/wangx86/ICLR-2026/ProteinWorkshop"
export DATA_PATH="/data/oliver_lab/wangx86/ps_data"

cd "$ROOT_DIR"

# GPU to use
GPU=0

# Create logs directory
mkdir -p "logs/baselines-125"

# Dataset configuration
DATASET="scop_proteinshake"
TASK="multiclass_graph_classification"
DATASET_SHORT="scop"

# Splits (deterministic)
SPLITS=("random")

# Models and their features (lookup) — gvp removed
declare -A MODEL_FEATURES=(
  [gear_net]="all_invariant_ca"
  [gear_net_edge]="all_invariant_ca"
)

# Deterministic model iteration order — gvp removed
MODELS_ORDER=(gear_net gear_net_edge)

# Run experiments
for split in "${SPLITS[@]}"; do
  dataset_config="${DATASET_SHORT}_${split}"

  for model in "${MODELS_ORDER[@]}"; do
    # Skip if log already exists (meaning it was run)
    log_file="logs/baselines-125/${dataset_config}_${model}.log"
    if [[ -f "$log_file" ]]; then
      echo "Skipping: ${model} on ${dataset_config} (log exists)"
      continue
    fi

    # Safety check for MODEL_FEATURES
    if [[ -z ${MODEL_FEATURES[$model]+_} ]]; then
      echo "ERROR: MODEL_FEATURES missing entry for model '$model'" >&2
      exit 1
    fi

    features="${MODEL_FEATURES[$model]}"

    echo "Running: ${model} on ${dataset_config} (GPU ${GPU}) -> ${log_file}"

    CUDA_VISIBLE_DEVICES="${GPU}" python proteinworkshop/train.py \
      "dataset=${dataset_config}" \
      "task=${TASK}" \
      "encoder=${model}" \
      "features=${features}" \
      "test=True" \
      "hydra.run.dir=outputs/train-125/${dataset_config}/${model}" \
      "trainer.max_epochs=100" \
      >"${log_file}" 2>&1

    echo "Completed: ${model} on ${dataset_config}"
  done
done

echo "All experiments completed!"

