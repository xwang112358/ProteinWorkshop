#!/usr/bin/env bash
set -euo pipefail

# GPU to use
GPU=7

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

# Run experiments for GO dataset on both splits
for split in "${SPLITS[@]}"; do
  dataset_config="go_${split}"

  for model in "${MODELS_ORDER[@]}"; do
    features="${MODEL_FEATURES[$model]}"
    log_file="logs/baselines-go/${dataset_config}_${model}.log"

    echo "Running: ${model} on ${dataset_config} (GPU ${GPU}) -> ${log_file}"

    CUDA_VISIBLE_DEVICES="${GPU}" python proteinworkshop/train.py \
      "dataset=${dataset_config}" \
      "task=${TASK}" \
      "encoder=${model}" \
      "features=${features}" \
      "test=True" \
      "hydra.run.dir=outputs/train-go/${dataset_config}/${model}" \
      >"${log_file}" 2>&1
  done
done

echo "All GO dataset experiments completed!"
