#!/bin/bash

################################################################################
# Example Configuration File for run_experiments.sh
# 
# Copy this file and modify the variables to customize your experiments.
# This example shows different configuration options.
################################################################################

################################################################################
# CONFIGURATION EXAMPLES
################################################################################

# Example 1: CLAP with cosine similarity, basic retrieval
ENCODER_NAME="clap"
MODEL_NAME="laion/clap-htsat-unfused"
DISTANCE_METRIC="cosine"
RETRIEVAL_STRATEGY="basic"
K_NEIGHBORS=5

# Example 2: Wav2Vec2 with L2 distance, rerank strategy
# ENCODER_NAME="wav2vec2"
# MODEL_NAME="facebook/wav2vec2-base"
# DISTANCE_METRIC="l2"
# RETRIEVAL_STRATEGY="rerank"
# K_NEIGHBORS=10

# Example 3: HuBERT with inner product, basic retrieval
# ENCODER_NAME="hubert"
# MODEL_NAME="facebook/hubert-base-ls960"
# DISTANCE_METRIC="ip"
# RETRIEVAL_STRATEGY="basic"
# K_NEIGHBORS=7

# Paths Configuration
# For Docker container:
BASE_DIR="/app"
# For local execution:
# BASE_DIR="."

CHROMADB_PATH="${BASE_DIR}/chromadb"
LOG_DIR="${BASE_DIR}/logs"
SCRIPTS_DIR="${BASE_DIR}/scripts"

# Dataset Paths
# Adjust these paths based on your environment
DATASET_BASE_PATH="${BASE_DIR}/data"

# Example: Run only specific datasets
# DATASETS=(
#     "esc50:${DATASET_BASE_PATH}/ESC-50"
#     "gtzan:${DATASET_BASE_PATH}/GTZAN"
# )

# Example: Run all datasets including Hugging Face
DATASETS=(
    "esc50:${DATASET_BASE_PATH}/ESC-50"
    "urbansound8k:${DATASET_BASE_PATH}/UrbanSound8K"
    "gtzan:${DATASET_BASE_PATH}/GTZAN"
    "nsynth:hf:jg583/NSynth"
)

# Experiment Settings
BATCH_SIZE=100
USE_GPU=true
SKIP_INDEXING=false  # Set to true if ChromaDB already indexed
SKIP_EVALUATION=false  # Set to true to only index, don't evaluate

# Logging
VERBOSE=true
SAVE_CONFUSION_MATRIX=true

################################################################################
# USAGE EXAMPLES
################################################################################

# 1. Run all experiments with default settings:
#    ./run_experiments.sh

# 2. Run only indexing (skip evaluation):
#    Set SKIP_EVALUATION=true in this file, then:
#    ./run_experiments.sh

# 3. Run only evaluation (skip indexing):
#    Set SKIP_INDEXING=true in this file, then:
#    ./run_experiments.sh

# 4. Compare different encoders:
#    Create multiple config files (e.g., config_clap.sh, config_wav2vec2.sh)
#    Source each config and run:
#    source config_clap.sh && ./run_experiments.sh
#    source config_wav2vec2.sh && ./run_experiments.sh

# 5. Compare different retrieval strategies:
#    Run with RETRIEVAL_STRATEGY="basic", then change to "rerank" and run again

################################################################################
# ADVANCED: Multiple Experiment Configurations
################################################################################

# You can create separate config files for different experiment scenarios:

# config_clap_cosine_basic.sh:
#   ENCODER_NAME="clap"
#   DISTANCE_METRIC="cosine"
#   RETRIEVAL_STRATEGY="basic"

# config_clap_cosine_rerank.sh:
#   ENCODER_NAME="clap"
#   DISTANCE_METRIC="cosine"
#   RETRIEVAL_STRATEGY="rerank"

# config_wav2vec2_l2_basic.sh:
#   ENCODER_NAME="wav2vec2"
#   DISTANCE_METRIC="l2"
#   RETRIEVAL_STRATEGY="basic"

# Then run:
#   source config_clap_cosine_basic.sh && ./run_experiments.sh
#   source config_clap_cosine_rerank.sh && ./run_experiments.sh
#   source config_wav2vec2_l2_basic.sh && ./run_experiments.sh

