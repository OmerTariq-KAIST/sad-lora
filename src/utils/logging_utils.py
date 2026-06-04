"""Logging utilities for SAD-LoRA training and spectral analysis."""

import logging
import os
from typing import Any

import torch


logger = logging.getLogger("sad_lora")


def setup_logging(
    project_name: str = "sad-lora",
    run_name: str | None = None,
    use_wandb: bool = True,
    log_level: int = logging.INFO,
) -> Any:
    """Initialize logging backends.

    Args:
        project_name: W&B project name.
        run_name: W&B run name (auto-generated if None).
        use_wandb: Whether to initialize W&B.
        log_level: Python logging level.

    Returns:
        W&B run object if use_wandb, else None.
    """
    logging.basicConfig(
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        level=log_level,
    )

    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=project_name, name=run_name)
        except ImportError:
            logger.warning("wandb not installed, falling back to console logging")

    return wandb_run


def log_metrics(metrics: dict[str, float], step: int, wandb_run: Any = None) -> None:
    """Log metrics to console and optionally W&B."""
    logger.info(
        "step=%d | %s",
        step,
        " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
    )
    if wandb_run is not None:
        import wandb
        wandb.log(metrics, step=step)


def log_spectral_snapshot(
    layer_metrics: dict[str, dict[str, Any]],
    step: int,
    save_dir: str | None = None,
    wandb_run: Any = None,
) -> None:
    """Log detailed per-layer spectral metrics.

    Args:
        layer_metrics: {layer_name: {metric_name: value}}.
        step: Current training step.
        save_dir: If provided, save tensors to disk.
        wandb_run: W&B run for logging.
    """
    for layer_name, metrics in layer_metrics.items():
        prefix = f"spectral/{layer_name}"
        flat = {f"{prefix}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}

        if wandb_run is not None:
            import wandb
            wandb.log(flat, step=step)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"spectral_step_{step}.pt")
        torch.save(layer_metrics, path)
        logger.info("Saved spectral snapshot to %s", path)
