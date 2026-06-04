"""Forgetting evaluation metrics.

Evaluates model on pretraining-related benchmarks to measure
how much pretrained knowledge is preserved after LoRA fine-tuning.

Hypothesis: SAD-LoRA reduces forgetting because fewer intruder
dimensions means less interference with pretrained knowledge.
"""

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger("sad_lora.evaluation")


class ForgettingEvaluator:
    """Evaluates forgetting on pretraining benchmarks.

    Supports:
    - WikiText-2 perplexity (language modeling)
    - LAMBADA accuracy (cloze completion)
    - BoolQ accuracy (general knowledge)
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    @torch.no_grad()
    def evaluate_perplexity(
        self, model: Any, eval_loader: DataLoader, max_batches: int = 100
    ) -> float:
        """Compute perplexity on a language modeling dataset.

        Args:
            model: Causal LM model.
            eval_loader: DataLoader with 'input_ids'.
            max_batches: Cap on number of batches.

        Returns:
            Perplexity (lower is better).
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        for i, batch in enumerate(eval_loader):
            if i >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", torch.ones_like(input_ids))
            attention_mask = attention_mask.to(self.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            # Shift for causal LM: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            )
            loss = loss.view(shift_labels.shape)
            masked_loss = (loss * shift_mask).sum()
            n_tokens = shift_mask.sum()

            total_loss += masked_loss.item()
            total_tokens += n_tokens.item()

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = float(torch.exp(torch.tensor(avg_loss)).item())
        logger.info("Perplexity: %.2f (avg_loss=%.4f, tokens=%d)", ppl, avg_loss, total_tokens)
        return ppl

    @torch.no_grad()
    def evaluate_accuracy(
        self, model: Any, eval_loader: DataLoader, max_batches: int = 200
    ) -> float:
        """Evaluate classification accuracy on a benchmark.

        Works for LAMBADA (predict last word) and BoolQ (binary classification).

        Args:
            model: Model with .forward(**batch) returning logits.
            eval_loader: DataLoader with 'input_ids' and 'labels'.
            max_batches: Cap on number of batches.

        Returns:
            Accuracy in [0, 1].
        """
        model.eval()
        correct = 0
        total = 0

        for i, batch in enumerate(eval_loader):
            if i >= max_batches:
                break

            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            labels = batch.pop("labels", None)
            if labels is None:
                continue

            outputs = model(**batch)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            if logits.dim() == 2:
                preds = logits.argmax(dim=-1)
            else:
                # For LM-style tasks, use the last token prediction
                preds = logits[:, -1, :].argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.shape[0]

        acc = correct / max(total, 1)
        logger.info("Accuracy: %.4f (%d/%d)", acc, correct, total)
        return acc

    def evaluate_all(
        self,
        model: Any,
        dataloaders: dict[str, DataLoader],
    ) -> dict[str, float]:
        """Run all forgetting benchmarks.

        Args:
            model: The model to evaluate.
            dataloaders: {benchmark_name: DataLoader}.

        Returns:
            {benchmark_name: metric_value}.
        """
        results = {}
        for name, loader in dataloaders.items():
            if "wikitext" in name.lower():
                results[name] = self.evaluate_perplexity(model, loader)
            else:
                results[name] = self.evaluate_accuracy(model, loader)

        return results
