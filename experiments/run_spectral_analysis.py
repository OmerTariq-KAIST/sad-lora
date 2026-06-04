"""Experiment 3: Spectral Analysis of Trained Adapters.

Runs on checkpoints from Experiment 2 to produce:
- Figure 3: Spectral overlap heatmaps (standard KD vs SAD-LoRA)
- Table 2: Aggregate spectral metrics (alignment, intruder dims, forgetting)
- Error decomposition bar charts (stacked Term I/II/III)

Can also be used standalone for Phase 1-2 offline spectral analysis.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader

from src.spectral_analysis import TeacherSpectralAnalyzer, DataWeightedSubspaceEstimator
from src.models import SADLoRAModel
from src.evaluation import SpectralMetrics

logger = logging.getLogger("sad_lora.exp_spectral")


def analyze_checkpoint(
    checkpoint_dir: str,
    teacher_model_name: str,
    student_model_name: str,
    teacher_checkpoint: str | None,
    task_name: str,
    rank: int,
    target_modules: list[str],
    device: str = "cuda",
) -> dict:
    """Run full spectral analysis on a trained checkpoint.

    Returns comprehensive per-layer metrics including alignment,
    intruder counts, error decomposition, and effective rank.
    """
    num_labels = {"sst2": 2, "mrpc": 2, "stsb": 1, "cola": 2, "qnli": 2, "rte": 2}.get(task_name, 2)

    # Load models
    teacher = AutoModelForSequenceClassification.from_pretrained(
        teacher_checkpoint or teacher_model_name, num_labels=num_labels
    ).to(device)
    pretrained = AutoModelForSequenceClassification.from_pretrained(
        student_model_name, num_labels=num_labels
    )
    student = AutoModelForSequenceClassification.from_pretrained(
        student_model_name, num_labels=num_labels
    )

    layer_names = []
    for name, module in student.named_modules():
        if isinstance(module, torch.nn.Linear) and any(t in name for t in target_modules):
            layer_names.append(name)

    # Phase 1: Teacher SVD
    analyzer = TeacherSpectralAnalyzer(
        teacher_model=teacher,
        pretrained_model=pretrained,
        layer_names=layer_names,
        r_max=64,
        device=device,
    )
    spectral_cache = analyzer.analyze()

    # Build student with LoRA
    sad_model = SADLoRAModel(
        base_model=student,
        target_modules=target_modules,
        default_rank=rank,
    )

    # Load trained LoRA weights
    lora_path = os.path.join(checkpoint_dir, "lora_weights.pt")
    if os.path.exists(lora_path):
        lora_state = torch.load(lora_path, map_location=device, weights_only=True)
        for name, layer in sad_model.lora_layers.items():
            prefix = name.replace(".", "_")
            if f"{prefix}.lora_B" in lora_state:
                layer.lora_B.data.copy_(lora_state[f"{prefix}.lora_B"])
                layer.lora_A.data.copy_(lora_state[f"{prefix}.lora_A"])
    else:
        logger.warning("No lora_weights.pt found in %s", checkpoint_dir)

    # Phase 2: Target subspaces (for alignment reference)
    tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    # Placeholder calibration — in practice, use actual task data
    target_subspaces = {}
    for name, info in spectral_cache.items():
        from src.spectral_analysis.data_weighted_svd import TargetSubspace
        r = min(rank, info.Sigma_T.shape[0])
        target_subspaces[name] = TargetSubspace(
            layer_name=name,
            U_tilde=info.U_T[:, :r],
            sigma_tilde=info.Sigma_T[:r],
            r_star=r,
            spectral_gap=(
                info.Sigma_T[r - 1].item() / info.Sigma_T[r].item()
                if r < len(info.Sigma_T) else float("inf")
            ),
            energy_captured=info.cumulative_energy[r - 1].item() if r <= len(info.cumulative_energy) else 1.0,
        )

    sad_model.set_target_subspaces(target_subspaces)
    sad_model.to(device)

    # Run spectral metrics
    metrics_computer = SpectralMetrics(
        model=sad_model,
        teacher_spectral_cache=spectral_cache,
        target_subspaces=target_subspaces,
    )
    metrics = metrics_computer.compute_all_metrics()

    return metrics


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)

    if args.results_dir:
        # Analyze all checkpoints from Experiment 2
        results_path = Path(args.results_dir)
        all_metrics = {}

        for ckpt_dir in sorted(results_path.glob("*/checkpoint_final")):
            parent = ckpt_dir.parent.name
            logger.info("Analyzing %s", parent)

            # Parse experiment name: {task}_{method}_r{rank}_s{seed}
            parts = parent.rsplit("_", 3)
            if len(parts) < 4:
                continue

            try:
                metrics = analyze_checkpoint(
                    checkpoint_dir=str(ckpt_dir),
                    teacher_model_name=args.teacher,
                    student_model_name=args.student,
                    teacher_checkpoint=args.teacher_checkpoint,
                    task_name=parts[0],
                    rank=int(parts[2].replace("r", "")),
                    target_modules=["query", "value"],
                    device=args.device,
                )
                all_metrics[parent] = metrics
            except Exception as e:
                logger.error("Failed to analyze %s: %s", parent, e)

        # Save
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(all_metrics, os.path.join(args.output_dir, "spectral_analysis.pt"))
        logger.info("Saved spectral analysis for %d runs", len(all_metrics))

    else:
        # Run standalone Phase 1-2 for a single model pair
        logger.info("Running standalone spectral analysis")
        metrics = analyze_checkpoint(
            checkpoint_dir=args.checkpoint or ".",
            teacher_model_name=args.teacher,
            student_model_name=args.student,
            teacher_checkpoint=args.teacher_checkpoint,
            task_name=args.task,
            rank=args.rank,
            target_modules=["query", "value"],
            device=args.device,
        )
        print(json.dumps(metrics["aggregate"], indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAD-LoRA Spectral Analysis")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--teacher", type=str, default="roberta-large")
    parser.add_argument("--student", type=str, default="roberta-base")
    parser.add_argument("--teacher_checkpoint", type=str, default=None)
    parser.add_argument("--task", type=str, default="sst2")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./results/spectral_analysis")
    parser.add_argument("--device", type=str, default="cuda")
    main(parser.parse_args())
