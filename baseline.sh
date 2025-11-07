#!/bin/bash

# GPU IDs to use
GPUS=(7 6 5 4)

# Create logs directory
mkdir -p logs/baselines

# Datasets and their task types
declare -A DATASET_TASKS
DATASET_TASKS[go_proteinshake]="multilabel_graph_classification"
DATASET_TASKS[ec_proteinshake]="multiclass_graph_classification"
DATASET_TASKS[scop_proteinshake]="multiclass_graph_classification"

# Splits
SPLITS=("random" "structure")

# Models and their features
declare -A MODEL_FEATURES
MODEL_FEATURES[gvp]="all_equivariant_ca"
MODEL_FEATURES[gcpnet]="all_equivariant_ca"
MODEL_FEATURES[gear_net]="all_invariant_ca"

# Counter for GPU assignment
gpu_idx=0

# Run experiments
for dataset in "${!DATASET_TASKS[@]}"; do
    task=${DATASET_TASKS[$dataset]}
    # Remove _proteinshake suffix for dataset config name
    dataset_short="${dataset/_proteinshake/}"
    
    for split in "${SPLITS[@]}"; do
        dataset_config="${dataset_short}_${split}"
        dataset_full="${dataset}_${split}"
        
        for model in "${!MODEL_FEATURES[@]}"; do
            features=${MODEL_FEATURES[$model]}
            gpu=${GPUS[$gpu_idx]}
            
            log_file="logs/baselines/${dataset_config}_${model}.log"
            echo "Running: ${model} on ${dataset_config} (GPU ${gpu}) -> ${log_file}"
            
            CUDA_VISIBLE_DEVICES=${gpu} python proteinworkshop/train.py \
                dataset=${dataset_config} \
                task=${task} \
                encoder=${model} \
                features=${features} \
                test=True \
                trainer.max_epochs=1 \
                hydra.run.dir="outputs/train/${dataset_config}/${model}" \
                > "${log_file}" 2>&1 &
            
            # Move to next GPU
            gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} ))
            
            # Wait when all GPUs have jobs
            if (( gpu_idx == 0 )); then
                wait -n
            fi
        done
    done
done

# Wait for all background jobs to complete
wait
echo "All experiments completed!"

