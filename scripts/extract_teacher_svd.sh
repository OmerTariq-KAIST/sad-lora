#!/bin/bash
# Phase 1-2: Offline spectral analysis.
# Extracts teacher SVD and data-weighted subspaces.
# Run this ONCE before training.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

TEACHER=${1:-"roberta-large"}
STUDENT=${2:-"roberta-base"}
TASK=${3:-"sst2"}
CACHE_DIR=${4:-"/data/Omer/saved_checkpoints/sad_lora_results/spectral_cache/${TASK}"}

echo "Extracting teacher spectral information..."
echo "  Teacher: $TEACHER"
echo "  Student: $STUDENT"
echo "  Task: $TASK"
echo "  Cache: $CACHE_DIR"

python experiments/run_spectral_analysis.py \
    --teacher "$TEACHER" \
    --student "$STUDENT" \
    --task "$TASK" \
    --rank 64 \
    --output_dir "$CACHE_DIR" \
    --device cuda

echo "Done. Cache saved to $CACHE_DIR"
