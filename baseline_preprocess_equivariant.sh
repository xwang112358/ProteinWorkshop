#!/bin/bash

# GPU ID to use
GPU=7

# Create logs directory
mkdir -p logs/preprocess

# Datasets and their task types (only SCOP and GO)
declare -A DATASET_TASKS
DATASET_TASKS[go_proteinshake]="multilabel_graph_classification"
DATASET_TASKS[scop_proteinshake]="multiclass_graph_classification"
DATASET_TASKS[ec_proteinshake]="multiclass_graph_classification"

# Splits
SPLITS=("random" "structure")

# Model (only GVP)
MODEL="gvp"
FEATURES="all_equivariant_ca"

# Clean corrupted cached files before running
# echo "Cleaning potentially corrupted cache files..."
rm -rf proteinworkshop/data/go_proteinshake/processed/*
rm -rf proteinworkshop/data/scop_proteinshake/processed/*
rm -rf proteinworkshop/data/ec_proteinshake/processed/*

# Run experiments sequentially
for dataset in "${!DATASET_TASKS[@]}"; do
    task=${DATASET_TASKS[$dataset]}
    # Remove _proteinshake suffix for dataset config name
    dataset_short="${dataset/_proteinshake/}"
    
    for split in "${SPLITS[@]}"; do
        dataset_config="${dataset_short}_${split}"
        
        log_file="logs/preprocess/${dataset_config}_${MODEL}.log"
        echo "Running: ${MODEL} on ${dataset_config} (GPU ${GPU}) -> ${log_file}"
        
        CUDA_VISIBLE_DEVICES=${GPU} python proteinworkshop/train.py \
            dataset=${dataset_config} \
            task=${task} \
            encoder=${MODEL} \
            features=${FEATURES} \
            test=True \
            trainer.max_epochs=1 \
            dataset.datamodule.num_workers=0 \
            dataset.datamodule.overwrite=True \
            hydra.run.dir="outputs/train/${dataset_config}/${MODEL}" \
            > "${log_file}" 2>&1
    done
done

echo "All experiments completed!"