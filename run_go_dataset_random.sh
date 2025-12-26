#!/usr/bin/env bash
set -euo pipefail

# Export environment variables
export ROOT_DIR="/home/wangx86/ICLR-2026/ProteinWorkshop"
export DATA_PATH="/data/oliver_lab/wangx86/ps_data"

cd "$ROOT_DIR"

# GPU to use
GPU=0

# Dataset
DATASET="go_proteinshake"
TASK="multilabel_graph_classification"

# Split
SPLIT="random"
dataset_config="go_${SPLIT}"

# Models and their features
declare -A MODEL_FEATURES=(
  [gcpnet]="all_equivariant_ca"
  [gear_net]="all_invariant_ca"
  [gear_net_edge]="all_invariant_ca"
)

# Models order
MODELS_ORDER=(gcpnet gear_net gear_net_edge)

# Run experiments for GO dataset on random split
for model in "${MODELS_ORDER[@]}"; do
  features="${MODEL_FEATURES[$model]}"

  echo "Running: ${model} on ${dataset_config} (GPU ${GPU})"

  CUDA_VISIBLE_DEVICES="${GPU}" python proteinworkshop/train.py \
    "dataset=${dataset_config}" \
    "task=${TASK}" \
    "encoder=${model}" \
    "features=${features}" \
    "test=True" \
    "hydra.run.dir=outputs/train-go/${dataset_config}/${model}" \
    "trainer.max_epochs=60"

  echo "Completed: ${model} on ${dataset_config}"
done

echo "All GO random split experiments completed!"
