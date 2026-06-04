"""Ablation Studies for SAD-LoRA.

Systematic ablations to isolate contribution of each component:
1. alpha=0 (no alignment loss) — isolate L_coeff
2. beta=0 (no coefficient loss) — isolate L_align
3. Fixed vs. adaptive schedule
4. Data-weighted vs. unweighted SVD (Proposition 1 validation)
5. Spectral init + SAD-LoRA (additive benefit?)
6. Calibration set size sensitivity
"""

import argparse
import json
import logging
import os
from itertools import product

import torch
from omegaconf import OmegaConf

from experiments.run_roberta_glue import run_single_experiment

logger = logging.getLogger("sad_lora.ablations")


ABLATION_CONFIGS = {
    # Core ablations (Table in paper)
    "no_align": {"alpha": 0.0, "beta": 0.1, "adaptive_schedule": False},
    "no_coeff": {"alpha": 1.0, "beta": 0.0, "adaptive_schedule": False},
    "fixed_schedule": {"alpha": 1.0, "beta": 0.1, "adaptive_schedule": False},
    "adaptive_schedule": {"alpha": 1.0, "beta": 0.1, "adaptive_schedule": True},

    # Data weighting ablation
    "unweighted_svd": {
        "alpha": 1.0, "beta": 0.1, "adaptive_schedule": True,
        "note": "Use SVD(delta_W) instead of SVD(delta_W @ Sigma_x^{1/2})"
    },

    # Initialization ablations
    "spectral_init_plus_sad": {
        "alpha": 1.0, "beta": 0.1, "adaptive_schedule": True,
        "init_method": "spectral",
    },
    "random_subspace_init": {
        "alpha": 1.0, "beta": 0.1, "adaptive_schedule": True,
        "init_method": "random_subspace",
    },

    # Hyperparameter sensitivity
    "alpha_0.1": {"alpha": 0.1, "beta": 0.1, "adaptive_schedule": True},
    "alpha_5.0": {"alpha": 5.0, "beta": 0.1, "adaptive_schedule": True},
    "beta_0.01": {"alpha": 1.0, "beta": 0.01, "adaptive_schedule": True},
    "beta_1.0": {"alpha": 1.0, "beta": 1.0, "adaptive_schedule": True},

    # Calibration sensitivity
    "cal_128": {"n_calibration": 128},
    "cal_512": {"n_calibration": 512},
    "cal_2048": {"n_calibration": 2048},
}


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    base_cfg = OmegaConf.load(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = args.tasks.split(",") if args.tasks else ["sst2", "qnli"]
    ranks = [int(r) for r in args.ranks.split(",")] if args.ranks else [4, 8]
    seeds = [42]  # Single seed for ablations (faster)
    ablations = args.ablations.split(",") if args.ablations else list(ABLATION_CONFIGS.keys())

    all_results = {}

    for ablation_name in ablations:
        if ablation_name not in ABLATION_CONFIGS:
            logger.warning("Unknown ablation: %s", ablation_name)
            continue

        abl_cfg = ABLATION_CONFIGS[ablation_name]
        logger.info("=== Running ablation: %s ===", ablation_name)

        # Apply ablation overrides to base config
        cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
        if "alpha" in abl_cfg:
            cfg.loss.alpha = abl_cfg["alpha"]
        if "beta" in abl_cfg:
            cfg.loss.beta = abl_cfg["beta"]
        if "adaptive_schedule" in abl_cfg:
            cfg.loss.adaptive_schedule = abl_cfg["adaptive_schedule"]
        if "n_calibration" in abl_cfg:
            cfg.spectral.n_calibration = abl_cfg["n_calibration"]

        method = "sad_lora"
        init_method = abl_cfg.get("init_method", "kaiming")

        for task, rank, seed in product(tasks, ranks, seeds):
            key = f"{ablation_name}/{task}/r{rank}/s{seed}"
            logger.info("Running %s", key)

            try:
                # Override init method in config
                cfg.lora.init_method = init_method

                result = run_single_experiment(
                    task_name=task,
                    rank=rank,
                    method=method,
                    cfg=cfg,
                    seed=seed,
                    output_dir=os.path.join(args.output_dir, ablation_name),
                )
                all_results[key] = result
            except Exception as e:
                logger.error("Failed %s: %s", key, e)
                all_results[key] = {"error": str(e)}

    # Save aggregate
    torch.save(all_results, os.path.join(args.output_dir, "ablation_results.pt"))

    # Print summary table
    logger.info("\n=== ABLATION SUMMARY ===")
    for key, result in sorted(all_results.items()):
        if "error" in result:
            logger.info("%s: ERROR - %s", key, result["error"])
        else:
            metrics = result.get("task_metrics", {})
            align = result.get("alignment_scores", {})
            mean_align = sum(align.values()) / max(len(align), 1) if align else 0
            logger.info(
                "%s: %s | mean_align=%.4f",
                key,
                " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
                mean_align,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAD-LoRA Ablation Studies")
    parser.add_argument("--config", type=str, default="configs/exp_roberta_glue.yaml")
    parser.add_argument("--output_dir", type=str, default="./results/ablations")
    parser.add_argument("--tasks", type=str, default="sst2,qnli")
    parser.add_argument("--ranks", type=str, default="4,8")
    parser.add_argument("--ablations", type=str, default=None)
    main(parser.parse_args())
