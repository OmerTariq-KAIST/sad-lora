"""Task-specific evaluation metrics for GLUE, MMLU, MT-Bench."""

import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger("sad_lora.evaluation")

# GLUE task metric definitions
GLUE_METRICS = {
    "sst2": {"metric": "accuracy", "higher_is_better": True, "num_labels": 2},
    "mrpc": {"metric": "f1", "higher_is_better": True, "num_labels": 2},
    "stsb": {"metric": "spearman", "higher_is_better": True, "num_labels": 1},
    "cola": {"metric": "matthews", "higher_is_better": True, "num_labels": 2},
    "mnli": {"metric": "accuracy", "higher_is_better": True, "num_labels": 3},
    "qnli": {"metric": "accuracy", "higher_is_better": True, "num_labels": 2},
    "qqp": {"metric": "f1", "higher_is_better": True, "num_labels": 2},
    "rte": {"metric": "accuracy", "higher_is_better": True, "num_labels": 2},
}


class TaskEvaluator:
    """Evaluates model on downstream task metrics."""

    def __init__(self, task_name: str, device: str = "cuda"):
        self.task_name = task_name
        self.device = device
        self.metric_info = GLUE_METRICS.get(task_name, {"metric": "accuracy", "higher_is_better": True})

    @torch.no_grad()
    def evaluate(self, model: Any, eval_loader: DataLoader) -> dict[str, float]:
        """Run evaluation on the given data loader.

        Args:
            model: Model with forward(**batch) -> outputs with .logits.
            eval_loader: DataLoader yielding batches with 'labels'.

        Returns:
            Dict with metric name -> value.
        """
        model.eval()
        all_preds = []
        all_labels = []

        for batch in eval_loader:
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            labels = batch.pop("labels", None)
            if labels is None:
                continue

            outputs = model(**batch)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            if self.task_name == "stsb":
                preds = logits.squeeze(-1)
            else:
                preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()

        return self._compute_metrics(preds, labels)

    def _compute_metrics(self, preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        metric_name = self.metric_info["metric"]

        if metric_name == "accuracy":
            return {"accuracy": float((preds == labels).mean())}

        elif metric_name == "f1":
            return {"f1": self._f1_score(preds, labels)}

        elif metric_name == "matthews":
            return {"matthews": self._matthews_corrcoef(preds, labels)}

        elif metric_name == "spearman":
            return {"spearman": self._spearman_correlation(preds, labels)}

        return {"accuracy": float((preds == labels).mean())}

    @staticmethod
    def _f1_score(preds: np.ndarray, labels: np.ndarray) -> float:
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall < 1e-10:
            return 0.0
        return float(2 * precision * recall / (precision + recall))

    @staticmethod
    def _matthews_corrcoef(preds: np.ndarray, labels: np.ndarray) -> float:
        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        if denom < 1e-10:
            return 0.0
        return float((tp * tn - fp * fn) / denom)

    @staticmethod
    def _spearman_correlation(preds: np.ndarray, labels: np.ndarray) -> float:
        n = len(preds)
        if n < 2:
            return 0.0
        rank_p = np.argsort(np.argsort(preds)).astype(float)
        rank_l = np.argsort(np.argsort(labels)).astype(float)
        d = rank_p - rank_l
        return float(1.0 - 6.0 * (d ** 2).sum() / (n * (n ** 2 - 1)))
