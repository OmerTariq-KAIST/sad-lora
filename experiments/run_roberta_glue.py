"""Experiment 2: RoBERTa Knowledge Distillation on GLUE.

Main NLU benchmark comparison. Distills RoBERTa-large (teacher) into
RoBERTa-base + LoRA (student) across 6 GLUE tasks, 5 ranks, 7 methods,
and 3 seeds.

Produces:
- Table 1: Performance across 6 GLUE tasks at rank 4 and 8
- Figure 2: Performance vs. rank curves for representative tasks
"""

import argparse
import json
import logging
import os

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader

from src.spectral_analysis import TeacherSpectralAnalyzer, DataWeightedSubspaceEstimator
from src.losses import SADLoRALoss
from src.models import SADLoRAModel
from src.training import SADLoRATrainer, DistillationEngine
from src.training.callbacks import CheckpointCallback, SpectralAnalysisCallback
from src.evaluation import TaskEvaluator, SpectralMetrics

logger = logging.getLogger("sad_lora.exp_roberta")

GLUE_TASKS = ["sst2", "mrpc", "stsb", "cola", "qnli", "rte"]


def load_and_tokenize(task_name: str, tokenizer, max_length: int = 128):
    """Load GLUE dataset and tokenize."""
    dataset = load_dataset("glue", task_name)

    # GLUE task input column names
    task_keys = {
        "sst2": ("sentence", None),
        "mrpc": ("sentence1", "sentence2"),
        "stsb": ("sentence1", "sentence2"),
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
    }
    key1, key2 = task_keys[task_name]

    def tokenize_fn(examples):
        args = (examples[key1],) if key2 is None else (examples[key1], examples[key2])
        result = tokenizer(*args, truncation=True, max_length=max_length, padding="max_length")
        result["labels"] = examples["label"]
        return result

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset["train"].column_names)
    tokenized.set_format("torch")
    return tokenized


def get_target_modules(model_name: str) -> list[str]:
    """Determine which linear layer names to apply LoRA to."""
    if "roberta" in model_name.lower():
        return ["query", "value"]
    return ["q_proj", "v_proj"]


def resolve_layer_names(model, target_modules: list[str]) -> list[str]:
    """Find all fully-qualified layer names matching target module substrings."""
    names = []
    for full_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(t in full_name for t in target_modules):
                names.append(full_name)
    return names


def run_single_experiment(
    task_name: str,
    rank: int,
    method: str,
    cfg: OmegaConf,
    seed: int,
    output_dir: str,
) -> dict:
    """Run one (task, rank, method, seed) combination."""
    run_dir = os.path.join(output_dir, f"{task_name}_{method}_r{rank}_s{seed}")
    result_file = os.path.join(run_dir, "result.json")
    if os.path.exists(result_file):
        logger.info("Skipping (already complete): %s", run_dir)
        with open(result_file) as f:
            return json.load(f)

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        "=== %s | rank=%d | method=%s | seed=%d ===", task_name, rank, method, seed
    )

    # Load models
    num_labels = {"sst2": 2, "mrpc": 2, "stsb": 1, "cola": 2, "qnli": 2, "rte": 2}[task_name]

    # KD teacher (roberta-large): provides soft labels for L_KD
    teacher = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.teacher_checkpoint.format(task=task_name)
        if cfg.model.teacher_checkpoint
        else cfg.model.teacher_name,
        num_labels=num_labels,
    ).to(device)

    # Spectral teacher (roberta-base fine-tuned): same arch as student.
    # delta_W = W_spectral_teacher - W_spectral_pretrained lives in student's weight space.
    spectral_teacher_ckpt = cfg.model.get("spectral_teacher_checkpoint", None)
    spectral_teacher_name = cfg.model.get("spectral_teacher_name", cfg.model.student_name)
    spectral_teacher = AutoModelForSequenceClassification.from_pretrained(
        spectral_teacher_ckpt.format(task=task_name) if spectral_teacher_ckpt else spectral_teacher_name,
        num_labels=num_labels,
    )
    # Pretrained baseline for spectral teacher (before task fine-tuning)
    spectral_pretrained = AutoModelForSequenceClassification.from_pretrained(
        spectral_teacher_name, num_labels=num_labels
    )

    student_base = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.student_name, num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.student_name)

    # Tokenize dataset
    datasets = load_and_tokenize(task_name, tokenizer, cfg.data.max_seq_length)
    train_loader = DataLoader(datasets["train"], batch_size=cfg.training.batch_size, shuffle=True)
    eval_loader = DataLoader(
        datasets["validation"],
        batch_size=cfg.training.batch_size * 2,
    )
    cal_loader = DataLoader(datasets["train"].select(range(min(1024, len(datasets["train"])))), batch_size=32)

    target_modules = get_target_modules(cfg.model.student_name)
    layer_names = resolve_layer_names(student_base, target_modules)

    # Phase 1-2: Spectral analysis (skip for methods that don't use it)
    target_subspaces = None
    if method in ("sad_lora", "sad_lora_no_coeff", "sad_lora_no_align", "neural_nuggets_init"):
        # Spectral analysis uses same-architecture models (roberta-base fine-tuned vs pretrained)
        # so that delta_W lives in student's weight space (768-dim), enabling direct alignment.
        spectral_analyzer = TeacherSpectralAnalyzer(
            teacher_model=spectral_teacher,
            pretrained_model=spectral_pretrained,
            layer_names=layer_names,
            r_max=64,
            device=device,
        )
        spectral_cache = spectral_analyzer.analyze()

        subspace_estimator = DataWeightedSubspaceEstimator(
            spectral_cache=spectral_cache,
            student_model=spectral_teacher,
            layer_names=layer_names,
            n_calibration=1024,
            energy_threshold=cfg.lora.auto_rank_energy_threshold,
            r_max=64,
            min_rank=rank,
            max_rank=rank,
            device=device,
        )
        target_subspaces = subspace_estimator.estimate(cal_loader)

    # Determine init method.
    # SAD-LoRA methods use spectral init (B = U_tilde) so L_align starts near 0.
    # With kaiming init, L_align ≈ 0.998 from step 1, overwhelming L_KD (alpha=1).
    init_method = "kaiming"
    if method in ("neural_nuggets_init", "sad_lora", "sad_lora_no_coeff", "sad_lora_no_align"):
        init_method = "spectral"

    # Build SAD-LoRA model
    sad_model = SADLoRAModel(
        base_model=student_base,
        target_modules=target_modules,
        default_rank=rank,
        lora_dropout=cfg.lora.lora_dropout,
        init_method=init_method,
    )

    if target_subspaces is not None:
        sad_model.set_target_subspaces(target_subspaces)

    # PiSSA initialization: B = U_r, A = diag(S_r) @ V_r^T, base = W - W_r
    if method == "pissa_init":
        for name, layer in sad_model.lora_layers.items():
            W = layer.base_layer.weight.data.float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            r = layer.r
            layer.lora_B.data = U[:, :r].to(layer.base_layer.weight.dtype)
            layer.lora_A.data = (torch.diag(S[:r]) @ Vh[:r, :]).to(layer.base_layer.weight.dtype)
            W_r = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
            layer.base_layer.weight.data = (W - W_r).to(layer.base_layer.weight.dtype)

    # Configure loss based on method.
    # feature_kd_lora uses logit-MSE distillation (distinct from KL-div in standard_kd_lora).
    # True hidden-state KD is infeasible here due to dim mismatch (large=1024, base=768).
    method_cfg = {
        "standard_kd_lora": {"alpha": 0.0, "beta": 0.0, "kd_loss_type": cfg.loss.kd_loss_type},
        "feature_kd_lora":  {"alpha": 0.0, "beta": 0.0, "kd_loss_type": "mse"},
        "neural_nuggets_init": {"alpha": 0.0, "beta": 0.0, "kd_loss_type": cfg.loss.kd_loss_type},
        "pissa_init":        {"alpha": 0.0, "beta": 0.0, "kd_loss_type": cfg.loss.kd_loss_type},
        "sad_lora":          {"alpha": cfg.loss.alpha, "beta": cfg.loss.beta, "kd_loss_type": cfg.loss.kd_loss_type},
        "sad_lora_no_coeff": {"alpha": cfg.loss.alpha, "beta": 0.0, "kd_loss_type": cfg.loss.kd_loss_type},
        "sad_lora_no_align": {"alpha": 0.0, "beta": cfg.loss.beta, "kd_loss_type": cfg.loss.kd_loss_type},
    }
    m_cfg = method_cfg.get(method, {"alpha": 0.0, "beta": 0.0, "kd_loss_type": cfg.loss.kd_loss_type})

    loss_fn = SADLoRALoss(
        alpha=m_cfg["alpha"],
        beta=m_cfg["beta"],
        temperature=cfg.loss.temperature,
        kd_loss_type=m_cfg["kd_loss_type"],
        coeff_loss_type=cfg.loss.coeff_loss_type,
        adaptive_schedule=cfg.loss.adaptive_schedule and m_cfg["alpha"] > 0,
    )

    # Distillation engine
    distill_engine = DistillationEngine(
        teacher_model=teacher,
        device=device,
        fp16_teacher=cfg.training.fp16,
    )

    # Callbacks
    callbacks = [
        CheckpointCallback(save_dir=run_dir, save_steps=cfg.evaluation.eval_steps),
        SpectralAnalysisCallback(
            analysis_steps=cfg.evaluation.spectral_analysis_steps,
            teacher_U_full={
                n: spectral_cache[n].U_T for n in layer_names
            } if target_subspaces is not None else None,
        ),
    ]

    # Train
    trainer = SADLoRATrainer(
        student_model=sad_model,
        distillation_engine=distill_engine,
        loss_fn=loss_fn,
        train_loader=train_loader,
        eval_loader=eval_loader,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        num_epochs=cfg.training.num_epochs,
        fp16=cfg.training.fp16,
        eval_steps=cfg.evaluation.eval_steps,
        callbacks=callbacks,
        device=device,
    )
    train_result = trainer.train()

    # Evaluate on task metric
    evaluator = TaskEvaluator(task_name=task_name, device=device)
    task_metrics = evaluator.evaluate(sad_model, eval_loader)

    # Spectral analysis
    spectral_analysis = None
    if target_subspaces is not None:
        spectral_eval = SpectralMetrics(
            model=sad_model,
            teacher_spectral_cache=spectral_cache,
            target_subspaces=target_subspaces,
        )
        spectral_analysis = spectral_eval.compute_all_metrics()

    result = {
        "task": task_name,
        "method": method,
        "rank": rank,
        "seed": seed,
        "task_metrics": task_metrics,
        "training_result": {
            "final_metrics": train_result["final_metrics"],
            "total_steps": train_result["total_steps"],
        },
        "spectral_analysis": spectral_analysis,
        "alignment_scores": sad_model.get_all_alignment_scores(),
    }

    # Save individual result
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(
            {k: v for k, v in result.items() if k != "spectral_analysis"},
            f, indent=2, default=str,
        )

    logger.info(
        "Result: %s | %s | r=%d | %s",
        task_name, method, rank,
        " | ".join(f"{k}={v:.4f}" for k, v in task_metrics.items()),
    )

    return result


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = OmegaConf.load(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = cfg.data.get("tasks", GLUE_TASKS)
    ranks = cfg.lora.get("ranks_to_test", [1, 2, 4, 8, 16])
    seeds = cfg.training.get("seeds", [42, 43, 44])
    methods = [m["name"] for m in cfg.get("methods", [{"name": "sad_lora"}])]

    all_results = []
    for task in tasks:
        for rank in ranks:
            for method in methods:
                for seed in seeds:
                    result = run_single_experiment(
                        task_name=task,
                        rank=rank,
                        method=method,
                        cfg=cfg,
                        seed=seed,
                        output_dir=args.output_dir,
                    )
                    all_results.append(result)

    # Save aggregate results
    torch.save(all_results, os.path.join(args.output_dir, "all_results.pt"))
    logger.info("All experiments complete. Results saved to %s", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAD-LoRA RoBERTa GLUE Experiment")
    parser.add_argument("--config", type=str, default="configs/exp_roberta_glue.yaml")
    parser.add_argument("--output_dir", type=str, default="./results/roberta_glue")
    main(parser.parse_args())
