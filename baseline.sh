#!/usr/bin/env bash
set -euo pipefail

# GPU IDs to use (ordered) — only GPUs 6 and 7
GPUS=(6 7)

# Create logs directory
mkdir -p "logs/baselines-117"

# Datasets and their task types (lookup)
declare -A DATASET_TASKS=(
  [go_proteinshake]="multilabel_graph_classification"
  [ec_proteinshake]="multiclass_graph_classification"
  [scop_proteinshake]="multiclass_graph_classification"
)

# Deterministic dataset iteration order: ec -> scop -> go
DATASETS_ORDER=(ec_proteinshake scop_proteinshake go_proteinshake)

# Splits (deterministic)
SPLITS=("random")

# Models and their features (lookup) — gvp removed
declare -A MODEL_FEATURES=(
  [gcpnet]="all_equivariant_ca"
  [gear_net]="all_invariant_ca"
  [gearnet_edge]="all_invariant_ca"
)

# Deterministic model iteration order — gvp removed
MODELS_ORDER=(gcpnet gear_net gearnet_edge)

# Counter for GPU assignment
gpu_idx=0

# Run experiments
for dataset in "${DATASETS_ORDER[@]}"; do
  task="${DATASET_TASKS[$dataset]}"
  # Remove _proteinshake suffix for dataset config name
  dataset_short="${dataset/_proteinshake/}"

  for split in "${SPLITS[@]}"; do
    dataset_config="${dataset_short}_${split}"

    for model in "${MODELS_ORDER[@]}"; do
      features="${MODEL_FEATURES[$model]}"
      gpu="${GPUS[$gpu_idx]}"

      log_file="logs/baselines-117/${dataset_config}_${model}.log"
      echo "Running: ${model} on ${dataset_config} (GPU ${gpu}) -> ${log_file}"

      CUDA_VISIBLE_DEVICES="${gpu}" python proteinworkshop/train.py \
        "dataset=${dataset_config}" \
        "task=${task}" \
        "encoder=${model}" \
        "features=${features}" \
        "test=True" \
        "hydra.run.dir=outputs/train-117/${dataset_config}/${model}" \
        >"${log_file}" 2>&1 &

      # Move to next GPU (max parallelism = number of GPUs = 2)
      gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} ))

      # When we've dispatched to all GPUs, wait for any one job to finish
      if (( gpu_idx == 0 )); then
        wait -n
      fi
    done
  done
done

# Wait for all background jobs to complete
wait
echo "All experiments completed!"
