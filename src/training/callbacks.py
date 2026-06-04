"""Training callbacks for logging, checkpointing, and spectral analysis."""

import logging
import os
from typing import Any

import torch

logger = logging.getLogger("sad_lora.callbacks")


class CheckpointCallback:
    """Saves model checkpoints and spectral snapshots."""

    def __init__(
        self,
        save_dir: str,
        save_steps: int = 1000,
        save_spectral: bool = True,
    ):
        self.save_dir = save_dir
        self.save_steps = save_steps
        self.save_spectral = save_spectral
        os.makedirs(save_dir, exist_ok=True)

    def on_step_end(self, step: int, model: Any, metrics: dict) -> None:
        if step > 0 and step % self.save_steps == 0:
            self._save_checkpoint(step, model, metrics)

    def on_train_end(self, step: int, model: Any, metrics: dict) -> None:
        self._save_checkpoint(step, model, metrics, is_final=True)

    def _save_checkpoint(
        self, step: int, model: Any, metrics: dict, is_final: bool = False
    ) -> None:
        tag = "final" if is_final else f"step_{step}"
        path = os.path.join(self.save_dir, f"checkpoint_{tag}")
        os.makedirs(path, exist_ok=True)

        # Save LoRA weights
        lora_state = model.get_lora_state_dict()
        torch.save(lora_state, os.path.join(path, "lora_weights.pt"))

        # Save spectral diagnostics
        if self.save_spectral:
            spectral_data = {
                "alignment_scores": model.get_all_alignment_scores(),
                "step": step,
                "metrics": metrics,
            }
            torch.save(spectral_data, os.path.join(path, "spectral_data.pt"))

        logger.info("Saved checkpoint to %s", path)


class SpectralAnalysisCallback:
    """Runs periodic spectral analysis during training."""

    def __init__(
        self,
        analysis_steps: int = 1000,
        teacher_U_full: dict[str, torch.Tensor] | None = None,
        intruder_threshold: float = 0.1,
    ):
        """
        Args:
            analysis_steps: Run analysis every N steps.
            teacher_U_full: {layer_name: (d_out, k)} full teacher singular vectors
                for intruder dimension counting.
            intruder_threshold: Cosine similarity threshold for intruder detection.
        """
        self.analysis_steps = analysis_steps
        self.teacher_U_full = teacher_U_full
        self.intruder_threshold = intruder_threshold
        self.history: list[dict] = []

    def on_step_end(self, step: int, model: Any, metrics: dict) -> dict | None:
        if step == 0 or step % self.analysis_steps != 0:
            return None

        analysis = {"step": step}
        for name, layer in model.lora_layers.items():
            if not layer.target_is_set:
                continue

            layer_data = {
                "alignment_score": layer.get_alignment_score(),
                "principal_angles": layer.get_principal_angles().tolist(),
                "adapter_singular_values": layer.get_adapter_singular_values().tolist(),
                "target_singular_values": layer.sigma_target.tolist(),
            }

            if self.teacher_U_full is not None and name in self.teacher_U_full:
                layer_data["intruder_count"] = layer.get_intruder_dimension_count(
                    self.teacher_U_full[name].to(layer.lora_B.device),
                    threshold=self.intruder_threshold,
                )

            analysis[name] = layer_data

        self.history.append(analysis)
        logger.info(
            "Step %d spectral analysis: mean_alignment=%.4f",
            step,
            _mean_alignment(analysis),
        )
        return analysis


class EarlyStoppingCallback:
    """Stop training early if alignment plateaus."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self._best_loss: float | None = None
        self._counter = 0

    def on_step_end(self, step: int, model: Any, metrics: dict) -> bool:
        """Returns True if training should stop."""
        loss = metrics.get("total", float("inf"))

        if self._best_loss is None or loss < self._best_loss - self.min_delta:
            self._best_loss = loss
            self._counter = 0
        else:
            self._counter += 1

        return self._counter >= self.patience


def _mean_alignment(analysis: dict) -> float:
    scores = [
        v["alignment_score"]
        for k, v in analysis.items()
        if isinstance(v, dict) and "alignment_score" in v
    ]
    return sum(scores) / max(len(scores), 1)
