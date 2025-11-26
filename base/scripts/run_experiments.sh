#!/bin/bash

################################################################################
# Experiment Runner Script - Modular Configuration
# 
# This script runs indexing and evaluation experiments across multiple datasets
# using the CLAP model. All settings are configurable at the top of the file.
################################################################################

set -e  # Exit on error

################################################################################
# CONFIGURATION - Modify these variables to change experiment settings
################################################################################

# Model Configuration
ENCODER_NAME="clap"
MODEL_NAME="laion/clap-htsat-unfused"

# Retrieval Configuration
DISTANCE_METRIC="cosine"  # Options: cosine, l2, ip
RETRIEVAL_STRATEGY="basic"  # Options: basic, rerank
K_NEIGHBORS=5

# Paths Configuration
BASE_DIR="/app"  # Inside Docker container, or "." for local
CHROMADB_PATH="${BASE_DIR}/chromadb"
LOG_DIR="${BASE_DIR}/logs"
SCRIPTS_DIR="${BASE_DIR}/scripts"

# Dataset Paths (adjust for your environment)
# For Docker: use /app/data/DATASET_NAME
# For local: use relative or absolute paths
DATASET_BASE_PATH="${BASE_DIR}/data"

# Datasets to run experiments on
# Format: "dataset_name:dataset_path" or "hf:dataset_name" for Hugging Face
# Use empty string "" to skip a dataset
DATASETS=(
    "esc50:${DATASET_BASE_PATH}/ESC-50"
    "urbansound8k:${DATASET_BASE_PATH}/UrbanSound8K"
    "gtzan:${DATASET_BASE_PATH}/GTZAN"
    "nsynth:hf:jg583/NSynth"
)

# Experiment Settings
BATCH_SIZE=100
USE_GPU=true  # Set to false to use CPU
SKIP_INDEXING=false  # Set to true to skip indexing (use existing ChromaDB)
SKIP_EVALUATION=false  # Set to true to skip evaluation (only index)

# Logging
VERBOSE=true  # Set to false to reduce output
SAVE_CONFUSION_MATRIX=true  # Save confusion matrix images

################################################################################
# HELPER FUNCTIONS
################################################################################

# Print colored output
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

print_section() {
    echo ""
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
    echo ""
}

# Check if GPU is available
check_gpu() {
    if [ "$USE_GPU" = true ]; then
        if command -v nvidia-smi &> /dev/null; then
            print_info "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
            GPU_FLAG="--use_gpu"
        else
            print_warning "GPU requested but nvidia-smi not found. Using CPU."
            GPU_FLAG=""
        fi
    else
        print_info "Using CPU (GPU disabled in config)"
        GPU_FLAG=""
    fi
}

# Parse dataset string
parse_dataset() {
    local dataset_str="$1"
    if [[ "$dataset_str" == hf:* ]]; then
        # Hugging Face dataset
        DATASET_NAME="${dataset_str}"
        DATASET_PATH=""
    else
        # Local dataset
        IFS=':' read -r DATASET_NAME DATASET_PATH <<< "$dataset_str"
    fi
}

# Get collection name (sanitized for ChromaDB)
get_collection_name() {
    local dataset_name="$1"
    # Replace : and / with _ for ChromaDB collection names
    echo "${dataset_name}" | sed 's/[:/]/_/g'
}

################################################################################
# INDEXING FUNCTION
################################################################################

run_indexing() {
    local dataset_name="$1"
    local dataset_path="$2"
    
    print_section "Indexing Dataset: $dataset_name"
    
    local collection_name=$(get_collection_name "${dataset_name}_${ENCODER_NAME}_${DISTANCE_METRIC}")
    
    print_info "Collection name: $collection_name"
    print_info "Encoder: $ENCODER_NAME"
    print_info "Model: $MODEL_NAME"
    print_info "Distance metric: $DISTANCE_METRIC"
    
    # Build command
    local cmd="python ${SCRIPTS_DIR}/index_esc50.py"
    cmd="$cmd --dataset_name $dataset_name"
    cmd="$cmd --encoder_name $ENCODER_NAME"
    cmd="$cmd --model_name $MODEL_NAME"
    cmd="$cmd --distance_metric $DISTANCE_METRIC"
    cmd="$cmd --chromadb_path $CHROMADB_PATH"
    cmd="$cmd --batch_size $BATCH_SIZE"
    
    # Add dataset_path only if not Hugging Face
    if [[ ! "$dataset_name" == hf:* ]] && [ -n "$dataset_path" ]; then
        cmd="$cmd --dataset_path $dataset_path"
    fi
    
    # Add GPU flag
    if [ -n "$GPU_FLAG" ]; then
        cmd="$cmd $GPU_FLAG"
    fi
    
    # Run indexing
    if [ "$VERBOSE" = true ]; then
        print_info "Running: $cmd"
    fi
    
    if eval "$cmd"; then
        print_success "Indexing completed for $dataset_name"
        return 0
    else
        print_error "Indexing failed for $dataset_name"
        return 1
    fi
}

################################################################################
# EVALUATION FUNCTION
################################################################################

run_evaluation() {
    local dataset_name="$1"
    local dataset_path="$2"
    
    print_section "Evaluating Dataset: $dataset_name"
    
    local collection_name=$(get_collection_name "${dataset_name}_${ENCODER_NAME}_${DISTANCE_METRIC}")
    local experiment_log_dir="${LOG_DIR}/${dataset_name}_${ENCODER_NAME}_${DISTANCE_METRIC}_${RETRIEVAL_STRATEGY}"
    
    print_info "Collection name: $collection_name"
    print_info "Retrieval strategy: $RETRIEVAL_STRATEGY"
    print_info "K neighbors: $K_NEIGHBORS"
    print_info "Log directory: $experiment_log_dir"
    
    # Build command
    local cmd="python ${SCRIPTS_DIR}/run_experiment.py"
    cmd="$cmd --dataset_name $dataset_name"
    cmd="$cmd --encoder_name $ENCODER_NAME"
    cmd="$cmd --model_name $MODEL_NAME"
    cmd="$cmd --distance_metric $DISTANCE_METRIC"
    cmd="$cmd --retrieval_strategy $RETRIEVAL_STRATEGY"
    cmd="$cmd --k $K_NEIGHBORS"
    cmd="$cmd --chromadb_path $CHROMADB_PATH"
    cmd="$cmd --log_dir $experiment_log_dir"
    
    # Add dataset_path only if not Hugging Face
    if [[ ! "$dataset_name" == hf:* ]] && [ -n "$dataset_path" ]; then
        cmd="$cmd --dataset_path $dataset_path"
    fi
    
    # Add GPU flag
    if [ -n "$GPU_FLAG" ]; then
        cmd="$cmd $GPU_FLAG"
    fi
    
    # Add confusion matrix save path
    if [ "$SAVE_CONFUSION_MATRIX" = true ]; then
        local cm_path="${experiment_log_dir}/confusion_matrix.png"
        cmd="$cmd --save_confusion_matrix $cm_path"
    fi
    
    # Run evaluation
    if [ "$VERBOSE" = true ]; then
        print_info "Running: $cmd"
    fi
    
    if eval "$cmd"; then
        print_success "Evaluation completed for $dataset_name"
        return 0
    else
        print_error "Evaluation failed for $dataset_name"
        return 1
    fi
}

################################################################################
# MAIN EXPERIMENT LOOP
################################################################################

main() {
    print_section "Starting Experiment Runner"
    
    # Print configuration
    print_info "Configuration:"
    echo "  Encoder: $ENCODER_NAME"
    echo "  Model: $MODEL_NAME"
    echo "  Distance metric: $DISTANCE_METRIC"
    echo "  Retrieval strategy: $RETRIEVAL_STRATEGY"
    echo "  K neighbors: $K_NEIGHBORS"
    echo "  ChromaDB path: $CHROMADB_PATH"
    echo "  Log directory: $LOG_DIR"
    echo "  GPU enabled: $USE_GPU"
    echo ""
    
    # Check GPU
    check_gpu
    
    # Create directories
    mkdir -p "$CHROMADB_PATH"
    mkdir -p "$LOG_DIR"
    
    # Track results
    local total_datasets=0
    local successful_indexing=0
    local successful_evaluation=0
    local failed_datasets=()
    
    # Process each dataset
    for dataset_config in "${DATASETS[@]}"; do
        # Skip empty entries
        if [ -z "$dataset_config" ]; then
            continue
        fi
        
        total_datasets=$((total_datasets + 1))
        
        # Parse dataset
        parse_dataset "$dataset_config"
        
        print_section "Processing Dataset: $DATASET_NAME"
        
        # Indexing
        if [ "$SKIP_INDEXING" = false ]; then
            if run_indexing "$DATASET_NAME" "$DATASET_PATH"; then
                successful_indexing=$((successful_indexing + 1))
            else
                failed_datasets+=("$DATASET_NAME (indexing)")
                if [ "$SKIP_EVALUATION" = false ]; then
                    print_warning "Skipping evaluation due to indexing failure"
                    continue
                fi
            fi
        else
            print_info "Skipping indexing (SKIP_INDEXING=true)"
        fi
        
        # Evaluation
        if [ "$SKIP_EVALUATION" = false ]; then
            if run_evaluation "$DATASET_NAME" "$DATASET_PATH"; then
                successful_evaluation=$((successful_evaluation + 1))
            else
                failed_datasets+=("$DATASET_NAME (evaluation)")
            fi
        else
            print_info "Skipping evaluation (SKIP_EVALUATION=true)"
        fi
    done
    
    # Print summary
    print_section "Experiment Summary"
    echo "Total datasets processed: $total_datasets"
    if [ "$SKIP_INDEXING" = false ]; then
        echo "Successful indexing: $successful_indexing/$total_datasets"
    fi
    if [ "$SKIP_EVALUATION" = false ]; then
        echo "Successful evaluation: $successful_evaluation/$total_datasets"
    fi
    
    if [ ${#failed_datasets[@]} -gt 0 ]; then
        print_error "Failed datasets:"
        for failed in "${failed_datasets[@]}"; do
            echo "  - $failed"
        done
        exit 1
    else
        print_success "All experiments completed successfully!"
        echo ""
        print_info "View results with TensorBoard:"
        echo "  tensorboard --logdir $LOG_DIR"
    fi
}

################################################################################
# RUN MAIN FUNCTION
################################################################################

main "$@"

