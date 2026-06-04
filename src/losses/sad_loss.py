"""Combined SAD-LoRA Loss.

L_SAD = L_KD + alpha * L_align + beta * L_coeff

Orchestrates all three loss components with:
- Shared QR decomposition between L_align and L_coeff
- Mixed precision: float16 for L_KD, forced float32 for spectral losses
- Adaptive alpha/beta scheduling based on alignment progress
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .alignment_loss import SubspaceAlignmentLoss
from .coefficient_loss import SpectralCoefficientLoss
from .adaptive_schedule import AdaptiveSchedule


class SADLoRALoss(nn.Module):
    """Full SAD-LoRA objective: L_SAD = L_KD + alpha * L_align + beta * L_coeff.

    This module computes all three loss components efficiently, sharing the
    QR decomposition of B across L_align and L_coeff per layer.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        temperature: float = 4.0,
        kd_loss_type: str = "kl_div",
        coeff_loss_type: str = "mse",
        adaptive_schedule: bool = True,
        warmup_steps: int = 100,
        normalize_coeff_loss: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.kd_loss_type = kd_loss_type

        self.align_loss_fn = SubspaceAlignmentLoss(normalize_per_layer=True)
        self.coeff_loss_fn = SpectralCoefficientLoss(
            loss_type=coeff_loss_type,
            normalize_by_target=normalize_coeff_loss,
        )
        self.schedule = AdaptiveSchedule(
            alpha_0=alpha,
            beta_0=beta,
            enabled=adaptive_schedule,
            warmup_steps=warmup_steps,
        )

        self._step = 0

    def forward(
        self,
        logits_student: Tensor,
        logits_teacher: Tensor,
        lora_layers: dict[str, dict[str, Tensor]],
        target_subspaces: dict[str, dict[str, Tensor]],
        labels: Tensor | None = None,
        is_training: bool = True,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute the full SAD-LoRA loss.

        Args:
            logits_student: (batch, ..., vocab) student model outputs.
            logits_teacher: (batch, ..., vocab) teacher model outputs (detached).
            lora_layers: {layer_name: {'B': (d_out, r), 'A': (r, d_in)}}.
            target_subspaces: {layer_name: {'U_tilde': (d_out, r), 'sigma_tilde': (r,)}}.
            labels: Optional ground-truth labels for task loss mixing.

        Returns:
            total_loss: Scalar loss for backpropagation.
            loss_dict: Detailed breakdown for logging.
        """
        # === Component 1: Knowledge Distillation Loss ===
        L_KD = self._compute_kd_loss(logits_student, logits_teacher)

        # === Components 2 & 3: Spectral Losses (forced float32) ===
        # Disable autocast for numerical stability in QR/SVD operations
        with torch.amp.autocast("cuda", enabled=False):
            L_align_total = torch.tensor(0.0, device=logits_student.device)
            L_coeff_total = torch.tensor(0.0, device=logits_student.device)
            n_layers = 0
            per_layer_align = {}
            per_layer_coeff = {}

            for name, params in lora_layers.items():
                if name not in target_subspaces:
                    continue

                B = params["B"]  # (d_out, r)
                A = params["A"]  # (r, d_in)
                U_target = target_subspaces[name]["U_tilde"]
                sigma_target = target_subspaces[name]["sigma_tilde"]

                # Shared QR decomposition: B = Q_B @ R_B (computed once)
                Q_B, R_B = SubspaceAlignmentLoss.compute_qr(B)

                # L_align: uses Q_B from the shared QR (no second QR call)
                align_loss, align_score, _ = self.align_loss_fn.forward_from_qr(Q_B, U_target)

                # L_coeff: uses R_B from the same shared QR
                coeff_loss, sigma_student = self.coeff_loss_fn(R_B, A, sigma_target)

                L_align_total = L_align_total + align_loss
                L_coeff_total = L_coeff_total + coeff_loss
                n_layers += 1

                per_layer_align[name] = align_score.item()
                per_layer_coeff[name] = coeff_loss.item()

            if n_layers > 0:
                L_align_total = L_align_total / n_layers
                L_coeff_total = L_coeff_total / n_layers

        # === Adaptive Scheduling ===
        if is_training:
            if self._step == 0:
                self.schedule.record_initial_alignment(L_align_total.item())
            alpha_t, beta_t = self.schedule.get_weights(L_align_total.item(), self._step)
            self._step += 1
        else:
            # Eval: read current schedule weights without advancing the counter
            alpha_t, beta_t = self.schedule.get_weights(L_align_total.item(), self._step)

        # === Total Loss ===
        total_loss = L_KD + alpha_t * L_align_total + beta_t * L_coeff_total

        # === Optional Task Loss ===
        if labels is not None:
            # For classification: cross-entropy on student logits
            if logits_student.dim() == 2:
                if labels.is_floating_point():
                    # Regression task (e.g., STS-B): cast both to float32 to
                    # avoid dtype mismatch when logits_student is fp16 from autocast
                    task_loss = F.mse_loss(
                        logits_student.float().squeeze(-1), labels.float()
                    )
                else:
                    task_loss = F.cross_entropy(logits_student, labels)
            else:
                # For generation: shift and compute per-token CE
                shift_logits = logits_student[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                task_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            total_loss = total_loss + task_loss

        loss_dict = {
            "total": total_loss.item(),
            "kd": L_KD.item(),
            "align": L_align_total.item(),
            "coeff": L_coeff_total.item(),
            "alpha_t": alpha_t,
            "beta_t": beta_t,
            "mean_alignment_score": (
                sum(per_layer_align.values()) / max(len(per_layer_align), 1)
            ),
        }

        return total_loss, loss_dict

    def _compute_kd_loss(self, logits_s: Tensor, logits_t: Tensor) -> Tensor:
        """Knowledge distillation loss.

        For classification (num_labels > 1): KL(softmax(z_T/tau) || softmax(z_S/tau)) * tau^2
        For regression (num_labels == 1): MSE between teacher and student predictions.
        Softmax of a single logit is always 1.0, making KL-div identically zero for
        regression tasks, so we detect this case and switch to MSE.
        """
        logits_t = logits_t.detach()

        # Regression: single output — KL-div degenerates (softmax([x]) = 1 always)
        if logits_s.shape[-1] == 1:
            return F.mse_loss(logits_s.float().squeeze(-1), logits_t.float().squeeze(-1))

        tau = self.temperature

        if self.kd_loss_type == "kl_div":
            p_t = F.softmax(logits_t / tau, dim=-1)
            log_p_s = F.log_softmax(logits_s / tau, dim=-1)
            # KLDivLoss expects log-probs as input, probs as target
            loss = F.kl_div(log_p_s, p_t, reduction="batchmean") * (tau ** 2)
        elif self.kd_loss_type == "mse":
            loss = F.mse_loss(logits_s.float(), logits_t.float())
        else:
            raise ValueError(f"Unknown kd_loss_type: {self.kd_loss_type}")

        return loss

    def reset_step_counter(self) -> None:
        """Reset internal step counter (call at start of training)."""
        self._step = 0
        self.schedule._initial_align_loss = None
