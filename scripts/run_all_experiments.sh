#!/bin/bash
# Run the full SAD-LoRA experimental pipeline.
# Usage: bash scripts/run_all_experiments.sh [--skip-synthetic] [--skip-glue] [--skip-llama]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

SKIP_SYNTHETIC=false
SKIP_GLUE=false
SKIP_LLAMA=false

for arg in "$@"; do
    case $arg in
        --skip-synthetic) SKIP_SYNTHETIC=true ;;
        --skip-glue) SKIP_GLUE=true ;;
        --skip-llama) SKIP_LLAMA=true ;;
    esac
done

echo "========================================="
echo "SAD-LoRA Experiment Pipeline"
echo "========================================="

# Experiment 1: Synthetic verification (CPU-feasible)
if [ "$SKIP_SYNTHETIC" = false ]; then
    echo ""
    echo "[1/4] Running synthetic verification..."
    python experiments/run_synthetic.py \
        --d_out 256 --d_in 128 \
        --n_samples 10000 --n_steps 5000 \
        --output_dir /data/Omer/saved_checkpoints/sad_lora_results/synthetic
fi

# Experiment 2: RoBERTa GLUE (GPU required)
if [ "$SKIP_GLUE" = false ]; then
    echo ""
    echo "[2/4] Running RoBERTa GLUE experiments..."
    python experiments/run_roberta_glue.py \
        --config configs/exp_roberta_glue.yaml \
        --output_dir /data/Omer/saved_checkpoints/sad_lora_results/roberta_glue
fi

# Experiment 3: Spectral analysis (runs on Exp 2 checkpoints)
if [ "$SKIP_GLUE" = false ]; then
    echo ""
    echo "[3/4] Running spectral analysis..."
    python experiments/run_spectral_analysis.py \
        --results_dir /data/Omer/saved_checkpoints/sad_lora_results/roberta_glue \
        --teacher roberta-large \
        --student roberta-base \
        --output_dir /data/Omer/saved_checkpoints/sad_lora_results/spectral_analysis
fi

# Experiment 4: Llama instruction tuning (A100 GPU required)
if [ "$SKIP_LLAMA" = false ]; then
    echo ""
    echo "[4/4] Running Llama instruction tuning..."
    python experiments/run_llama_instruct.py \
        --config configs/exp_llama_instruction.yaml \
        --output_dir /data/Omer/saved_checkpoints/sad_lora_results/llama_instruct
fi

# Ablation studies
echo ""
echo "[Bonus] Running ablation studies..."
python experiments/run_ablations.py \
    --config configs/exp_roberta_glue.yaml \
    --output_dir /data/Omer/saved_checkpoints/sad_lora_results/ablations \
    --tasks sst2,qnli --ranks 4,8

echo ""
echo "========================================="
echo "All experiments complete!"
echo "Results: /data/Omer/saved_checkpoints/sad_lora_results/"
echo "========================================="
