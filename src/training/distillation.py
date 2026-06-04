"""Knowledge distillation engine for SAD-LoRA.

Handles teacher/student forward passes, optional teacher logit caching,
and top-k KD for generation tasks.
"""

import logging
import os
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger("sad_lora.distillation")


class DistillationEngine:
    """Manages teacher model inference and logit caching.

    For classification (GLUE): logits are (n, num_classes) — small, always live.
    For generation (Llama): logits are (n, seq_len, vocab) — huge. Use top-k caching.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        device: str = "cuda",
        fp16_teacher: bool = True,
        cache_dir: str | None = None,
        top_k_kd: int | None = None,
    ):
        """
        Args:
            teacher_model: Fully fine-tuned teacher (frozen, eval mode).
            device: Device for teacher inference.
            fp16_teacher: Run teacher in float16 for memory efficiency.
            cache_dir: If set, pre-cache teacher logits to disk.
            top_k_kd: For generation tasks, only cache top-k logit values/indices.
        """
        self.teacher_model = teacher_model
        self.device = device
        self.fp16_teacher = fp16_teacher
        self.cache_dir = cache_dir
        self.top_k_kd = top_k_kd

        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)

        if fp16_teacher:
            self.teacher_model.half()

        self._cached_logits: dict[int, Tensor] | None = None

    @torch.no_grad()
    def get_teacher_logits(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute or retrieve teacher logits for a batch.

        Args:
            batch: Dict with 'input_ids', 'attention_mask', etc.

        Returns:
            Teacher logits tensor, matching student logit shape.
        """
        input_kwargs = {
            k: v.to(self.device) for k, v in batch.items()
            if k in ("input_ids", "attention_mask", "token_type_ids")
        }

        with torch.amp.autocast("cuda", enabled=self.fp16_teacher):
            outputs = self.teacher_model(**input_kwargs)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        return logits.float()  # Always return float32 for KD loss

    def precache_teacher_logits(
        self, dataloader: DataLoader, save_path: str | None = None
    ) -> dict[int, Tensor]:
        """Pre-compute teacher logits for the entire dataset.

        For generation tasks with top_k_kd, stores only top-k values and indices.

        Args:
            dataloader: Training data loader.
            save_path: If provided, save cache to disk.

        Returns:
            Dict mapping batch index -> teacher logits tensor.
        """
        logger.info("Pre-caching teacher logits...")
        cache = {}

        for idx, batch in enumerate(tqdm(dataloader, desc="Caching teacher logits")):
            logits = self.get_teacher_logits(batch)

            if self.top_k_kd is not None:
                # Store only top-k for memory efficiency
                values, indices = logits.topk(self.top_k_kd, dim=-1)
                cache[idx] = {"values": values.cpu(), "indices": indices.cpu()}
            else:
                cache[idx] = logits.cpu()

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(cache, save_path)
            logger.info("Saved teacher logit cache to %s", save_path)

        self._cached_logits = cache
        return cache

    def reconstruct_logits_from_topk(
        self,
        topk_data: dict[str, Tensor],
        vocab_size: int,
        fill_value: float = -1e4,
    ) -> Tensor:
        """Reconstruct full logit tensor from cached top-k.

        Args:
            topk_data: {'values': (batch, seq, k), 'indices': (batch, seq, k)}.
            vocab_size: Full vocabulary size.
            fill_value: Value for non-top-k positions (large negative = ~0 prob).

        Returns:
            (batch, seq, vocab) reconstructed logits.
        """
        values = topk_data["values"]
        indices = topk_data["indices"]
        shape = values.shape[:-1] + (vocab_size,)

        logits = torch.full(shape, fill_value, dtype=values.dtype, device=values.device)
        logits.scatter_(-1, indices, values)
        return logits
