#!/usr/bin/env bash
set -euo pipefail

# Export environment variables
export ROOT_DIR="/home/wangx86/ICLR-2026/ProteinWorkshop"
export DATA_PATH="/data/oliver_lab/wangx86/ps_data"

cd "$ROOT_DIR"

# Example 1: Run training on EC dataset with gcpnet
# CUDA_VISIBLE_DEVICES=0 python proteinworkshop/train.py \
#   dataset=ec_random \
#   task=multiclass_graph_classification \
#   encoder=gcpnet \
#   features=all_equivariant_ca \
#   trainer.max_epochs=1 \
#   test=True \
#   hydra.run.dir=outputs/train-test/ec_random/gcpnet

# Example 2: Run training on GO dataset with gcpnet (uncomment to use)
CUDA_VISIBLE_DEVICES=0 python proteinworkshop/train.py \
  dataset=go_random \
  task=multilabel_graph_classification \
  encoder=gcpnet \
  features=all_equivariant_ca \
  trainer.max_epochs=1 \
  test=True \
  hydra.run.dir=outputs/train-test/go_random/gcpnet
