"""Main SAD-LoRA training loop (Algorithm 3).

Orchestrates:
- Teacher inference (frozen, optional pre-caching)
- Student forward pass with LoRA
- SAD-LoRA loss computation (L_KD + alpha*L_align + beta*L_coeff)
- Gradient clipping for SVD stability
- Adaptive alpha/beta scheduling
- Periodic evaluation and spectral analysis callbacks
"""

import logging
import time
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..losses.sad_loss import SADLoRALoss
from ..models.sad_lora_model import SADLoRAModel
from .distillation import DistillationEngine
from .callbacks import CheckpointCallback, SpectralAnalysisCallback

logger = logging.getLogger("sad_lora.trainer")


class SADLoRATrainer:
    """Implements Algorithm 3: the SAD-LoRA training loop.

    Handles the complete Phase 3 pipeline: optimizer setup, mixed precision,
    gradient clipping, loss computation, logging, and checkpointing.
    """

    def __init__(
        self,
        student_model: SADLoRAModel,
        distillation_engine: DistillationEngine,
        loss_fn: SADLoRALoss,
        train_loader: DataLoader,
        eval_loader: DataLoader | None = None,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        num_epochs: int = 10,
        max_steps: int | None = None,
        warmup_ratio: float = 0.06,
        lr_scheduler_type: str = "cosine",
        fp16: bool = True,
        gradient_accumulation_steps: int = 1,
        eval_steps: int = 500,
        log_steps: int = 50,
        callbacks: list[Any] | None = None,
        device: str = "cuda",
    ):
        self.student = student_model
        self.distill = distillation_engine
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.fp16 = fp16
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.eval_steps = eval_steps
        self.log_steps = log_steps
        self.callbacks = callbacks or []
        self.device = device

        # Compute total steps
        steps_per_epoch = len(train_loader) // gradient_accumulation_steps
        self.total_steps = max_steps or (num_epochs * steps_per_epoch)

        # Optimizer: only LoRA parameters
        self.optimizer = AdamW(
            self.student.lora_parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # LR scheduler with warmup
        warmup_steps = int(self.total_steps * warmup_ratio)
        self.scheduler = self._build_scheduler(lr_scheduler_type, warmup_steps)

        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler("cuda", enabled=fp16)

        self.global_step = 0   # counts optimizer steps
        self._microbatch = 0   # counts all forward/backward microbatches
        self.best_eval_metric = -float("inf")

    def _build_scheduler(self, sched_type: str, warmup_steps: int) -> Any:
        warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=max(warmup_steps, 1))

        if sched_type == "cosine":
            main = CosineAnnealingLR(self.optimizer, T_max=max(self.total_steps - warmup_steps, 1))
        elif sched_type == "linear":
            main = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=max(self.total_steps - warmup_steps, 1),
            )
        elif sched_type == "constant":
            main = LinearLR(self.optimizer, start_factor=1.0, total_iters=1)
        else:
            raise ValueError(f"Unknown scheduler: {sched_type}")

        return SequentialLR(self.optimizer, [warmup, main], milestones=[warmup_steps])

    def train(self) -> dict[str, Any]:
        """Execute the full training loop.

        Returns:
            Final metrics dict with training history.
        """
        logger.info(
            "Starting SAD-LoRA training: %d steps, %d LoRA params",
            self.total_steps, self.student.count_lora_parameters(),
        )

        self.student.to(self.device)
        self.student.train()
        self.loss_fn.reset_step_counter()

        # Precompute target params (frozen buffers, moved to device once)
        target_params = self.student.get_all_target_params()

        history = []
        t_start = time.time()
        accum_loss = 0.0
        accum_count = 0

        epoch = 0
        while self.global_step < self.total_steps:
            epoch += 1
            for batch in self.train_loader:
                if self.global_step >= self.total_steps:
                    break

                step_metrics, optimizer_stepped = self._training_step(batch, target_params)

                # Skip logging/eval/history on microbatch-only iterations;
                # global_step only advances on optimizer steps.
                if not optimizer_stepped:
                    continue

                accum_loss += step_metrics["total"]
                accum_count += 1

                # Logging (in optimizer-step units)
                if self.global_step % self.log_steps == 0:
                    avg_loss = accum_loss / max(accum_count, 1)
                    elapsed = time.time() - t_start
                    logger.info(
                        "step=%d/%d | loss=%.4f | kd=%.4f | align=%.4f | coeff=%.4f | "
                        "alpha=%.3f | beta=%.3f | lr=%.2e | elapsed=%.0fs",
                        self.global_step, self.total_steps,
                        avg_loss, step_metrics["kd"],
                        step_metrics["align"], step_metrics["coeff"],
                        step_metrics["alpha_t"], step_metrics["beta_t"],
                        self.optimizer.param_groups[0]["lr"],
                        elapsed,
                    )
                    accum_loss, accum_count = 0.0, 0

                # Evaluation (in optimizer-step units)
                if (
                    self.eval_loader is not None
                    and self.eval_steps > 0
                    and self.global_step % self.eval_steps == 0
                ):
                    eval_metrics = self.evaluate()
                    step_metrics.update(eval_metrics)

                # Callbacks
                for cb in self.callbacks:
                    if hasattr(cb, "on_step_end"):
                        result = cb.on_step_end(self.global_step, self.student, step_metrics)
                        if result is True:
                            logger.info("Early stopping triggered at step %d", self.global_step)
                            return self._finalize(history)

                history.append(step_metrics)

        return self._finalize(history)

    def _training_step(
        self,
        batch: dict[str, Any],
        target_params: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, float], bool]:
        """Single training step with gradient accumulation.

        Returns:
            loss_dict: Loss breakdown from this microbatch.
            optimizer_stepped: True when an optimizer step was taken.
        """
        batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # Teacher forward (no grad, optional fp16)
        teacher_logits = self.distill.get_teacher_logits(batch)

        # Student forward (with autocast for mixed precision)
        with torch.amp.autocast("cuda", enabled=self.fp16):
            student_out = self.student(**{
                k: v for k, v in batch.items()
                if k in ("input_ids", "attention_mask", "token_type_ids")
            })
            student_logits = (
                student_out.logits if hasattr(student_out, "logits") else student_out[0]
            )

        # Get current LoRA params for spectral losses
        lora_params = self.student.get_all_lora_params()
        labels = batch.get("labels")

        # Compute SAD-LoRA loss
        loss, loss_dict = self.loss_fn(
            logits_student=student_logits,
            logits_teacher=teacher_logits,
            lora_layers=lora_params,
            target_subspaces=target_params,
            labels=labels,
        )

        # Scale for gradient accumulation
        loss = loss / self.gradient_accumulation_steps

        # Backward with mixed precision
        self.scaler.scale(loss).backward()

        self._microbatch += 1

        # Optimizer step at accumulation boundary
        optimizer_stepped = False
        if self._microbatch % self.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.student.lora_parameters(), self.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.global_step += 1   # advances only on optimizer steps
            optimizer_stepped = True

        return loss_dict, optimizer_stepped

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation on the eval set."""
        self.student.eval()
        total_loss = 0.0
        n_batches = 0

        target_params = self.student.get_all_target_params()

        for batch in self.eval_loader:
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            teacher_logits = self.distill.get_teacher_logits(batch)

            student_out = self.student(**{
                k: v for k, v in batch.items()
                if k in ("input_ids", "attention_mask", "token_type_ids")
            })
            student_logits = (
                student_out.logits if hasattr(student_out, "logits") else student_out[0]
            )

            lora_params = self.student.get_all_lora_params()
            _, loss_dict = self.loss_fn(
                logits_student=student_logits,
                logits_teacher=teacher_logits,
                lora_layers=lora_params,
                target_subspaces=target_params,
                labels=batch.get("labels"),
                is_training=False,
            )
            total_loss += loss_dict["total"]
            n_batches += 1

        self.student.train()

        eval_loss = total_loss / max(n_batches, 1)
        scores = self.student.get_all_alignment_scores()
        mean_align = sum(scores.values()) / max(len(scores), 1)

        logger.info(
            "Eval: loss=%.4f | mean_alignment=%.4f", eval_loss, mean_align
        )

        return {"eval_loss": eval_loss, "eval_mean_alignment": mean_align}

    def _finalize(self, history: list) -> dict[str, Any]:
        """Run final callbacks and return results."""
        final_metrics = history[-1] if history else {}
        for cb in self.callbacks:
            if hasattr(cb, "on_train_end"):
                cb.on_train_end(self.global_step, self.student, final_metrics)

        return {
            "final_metrics": final_metrics,
            "total_steps": self.global_step,
            "history": history,
        }
