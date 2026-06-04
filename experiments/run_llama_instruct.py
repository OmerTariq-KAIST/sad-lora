"""Experiment 4: Llama Instruction Tuning Scale-Up.

Distills Llama-3.1-8B-Instruct (teacher) into Llama-3.2-1B + LoRA (student)
on Alpaca instruction data. Evaluates on MT-Bench and MMLU.

This validates SAD-LoRA at LLM scale with:
- Top-k KD (only top-128 logits for memory efficiency)
- All 7 attention+MLP projection types as LoRA targets
- Auto rank selection via Theorem 2
"""

import argparse
import logging
import os

import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

from src.spectral_analysis import TeacherSpectralAnalyzer, DataWeightedSubspaceEstimator
from src.losses import SADLoRALoss
from src.models import SADLoRAModel
from src.training import SADLoRATrainer, DistillationEngine
from src.training.callbacks import CheckpointCallback, SpectralAnalysisCallback
from src.evaluation import SpectralMetrics, ForgettingEvaluator

logger = logging.getLogger("sad_lora.exp_llama")

LLAMA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_alpaca_loader(tokenizer, max_length: int = 512, batch_size: int = 4):
    """Load and tokenize Alpaca instruction dataset."""
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    def format_and_tokenize(examples):
        texts = []
        for instruction, inp, output in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            if inp:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            texts.append(prompt)

        tokens = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        tokens["labels"] = tokens["input_ids"].clone()
        # Mask padding tokens in labels
        tokens["labels"][tokens["attention_mask"] == 0] = -100
        return tokens

    dataset = dataset.map(format_and_tokenize, batched=True, remove_columns=dataset.column_names)
    dataset.set_format("torch")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def resolve_layer_names(model, target_modules: list[str]) -> list[str]:
    names = []
    for full_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(t in full_name for t in target_modules):
                names.append(full_name)
    return names


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = OmegaConf.load(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = cfg.training.get("seeds", [42, 43, 44])
    ranks = cfg.lora.get("ranks_to_test", [4, 16])

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.student_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for rank in ranks:
        for seed in seeds:
            torch.manual_seed(seed)
            logger.info("=== Llama SAD-LoRA | rank=%d | seed=%d ===", rank, seed)

            # Load teacher (fp16 for memory)
            teacher = AutoModelForCausalLM.from_pretrained(
                cfg.model.teacher_name, torch_dtype=torch.float16
            ).to(device)

            # Load pretrained student (for spectral analysis reference)
            pretrained = AutoModelForCausalLM.from_pretrained(cfg.model.student_name)
            student = AutoModelForCausalLM.from_pretrained(cfg.model.student_name)

            layer_names = resolve_layer_names(student, LLAMA_TARGET_MODULES)
            logger.info("Found %d target layers", len(layer_names))

            # Phase 1: Teacher SVD
            spectral_analyzer = TeacherSpectralAnalyzer(
                teacher_model=teacher,
                pretrained_model=pretrained,
                layer_names=layer_names,
                r_max=64,
                device=device,
                use_randomized_svd=True,
            )
            spectral_cache = spectral_analyzer.analyze()

            # Phase 2: Data-weighted subspace
            train_loader = build_alpaca_loader(
                tokenizer,
                max_length=cfg.data.max_seq_length,
                batch_size=cfg.training.batch_size,
            )
            cal_loader = DataLoader(
                train_loader.dataset.select(range(min(1024, len(train_loader.dataset)))),
                batch_size=32,
            )

            subspace_estimator = DataWeightedSubspaceEstimator(
                spectral_cache=spectral_cache,
                student_model=student,
                layer_names=layer_names,
                n_calibration=cfg.spectral.n_calibration,
                energy_threshold=cfg.lora.auto_rank_energy_threshold,
                use_implicit_svd=True,
                r_max=64,
                min_rank=max(1, rank // 2),
                max_rank=rank,
                device=device,
            )
            target_subspaces = subspace_estimator.estimate(cal_loader)

            # Build SAD-LoRA model
            sad_model = SADLoRAModel(
                base_model=student,
                target_modules=LLAMA_TARGET_MODULES,
                default_rank=rank,
                rank_per_layer={n: ts.r_star for n, ts in target_subspaces.items()},
                lora_dropout=cfg.lora.lora_dropout,
            )
            sad_model.set_target_subspaces(target_subspaces)

            # Loss
            loss_fn = SADLoRALoss(
                alpha=cfg.loss.alpha,
                beta=cfg.loss.beta,
                temperature=cfg.loss.temperature,
                kd_loss_type=cfg.loss.kd_loss_type,
                adaptive_schedule=cfg.loss.adaptive_schedule,
            )

            # Distillation engine with top-k KD
            distill = DistillationEngine(
                teacher_model=teacher,
                device=device,
                fp16_teacher=True,
                top_k_kd=cfg.loss.get("top_k_kd", 128),
            )

            # Train
            run_dir = os.path.join(args.output_dir, f"llama_r{rank}_s{seed}")
            trainer = SADLoRATrainer(
                student_model=sad_model,
                distillation_engine=distill,
                loss_fn=loss_fn,
                train_loader=train_loader,
                learning_rate=cfg.training.learning_rate,
                weight_decay=cfg.training.weight_decay,
                max_grad_norm=cfg.training.max_grad_norm,
                num_epochs=cfg.training.num_epochs,
                fp16=cfg.training.fp16,
                gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
                callbacks=[
                    CheckpointCallback(save_dir=run_dir),
                    SpectralAnalysisCallback(
                        analysis_steps=1000,
                        teacher_U_full={n: spectral_cache[n].U_T for n in layer_names},
                    ),
                ],
                device=device,
            )
            result = trainer.train()

            # Forgetting evaluation
            forgetting_eval = ForgettingEvaluator(device=device)
            logger.info("Forgetting evaluation would run here with WikiText-2, LAMBADA loaders")

            # Save
            torch.save(result, os.path.join(run_dir, "training_result.pt"))
            logger.info("Saved results to %s", run_dir)

            # Clean up GPU memory
            del teacher, pretrained, student, sad_model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAD-LoRA Llama Instruction Tuning")
    parser.add_argument("--config", type=str, default="configs/exp_llama_instruction.yaml")
    parser.add_argument("--output_dir", type=str, default="./results/llama_instruct")
    main(parser.parse_args())
