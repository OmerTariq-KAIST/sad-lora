# SAD-LoRA: Spectral Alignment Distillation for Low-Rank Adaptation

Official implementation of **SAD-LoRA**, accepted at ICML 2026.

> **SAD-LoRA: Spectral Alignment for Low-Rank Knowledge Distillation**  
> Omer Tariq  
> *ICML 2026 — CoLoRAI Workshop (Compression, LoRA, and Representation Alignment)*

SAD-LoRA distills a teacher model into a LoRA-adapted student by aligning the adapter's column space with the **data-weighted spectral subspace** of the teacher update. This eliminates the dominant source of low-rank distillation error — subspace misalignment — without changing the student architecture or adding inference-time overhead.


## Architecture
<img width="5632" height="2872" alt="SAD-LoRA" src="https://github.com/user-attachments/assets/3bf7223b-daf1-4878-be3f-5ba822f9522c" />



---

## Method

The teacher's task update ΔW_T is projected by the downstream input covariance:

```
W̃_T = ΔW_T · Σ_x^{1/2}
```

The leading left singular vectors of W̃_T define the **target subspace** for the LoRA adapter. SAD-LoRA enforces alignment throughout training via:

```
L_SAD = L_KD + α · L_align + β · L_coeff
```

- **L_align** — principal-angle discrepancy between colspan(B) and the teacher subspace (Grassmannian loss)
- **L_coeff** — singular value matching within the aligned subspace (optional)
- **L_KD** — standard logit distillation

## Key Results (RoBERTa-large → RoBERTa-base, GLUE)

| Task | Metric | KD-LoRA (r=8) | SAD-LoRA-Align (r=8) | Δ |
|------|--------|--------------|----------------------|---|
| STS-B | Spearman ρ | 0.847 | **0.893** | +0.046 |
| CoLA | MCC | 0.478 | **0.562** | +0.084 |
| QNLI | Acc | 92.0 | **92.8** | +0.8 |
| RTE | Acc | 67.4 | **72.2** | +4.8 |
| SST-2 | Acc | 93.7 | 93.6 | ≈ |
| MRPC | F1 | 90.2 | 90.1 | ≈ |

SAD-LoRA-Align (alignment loss only, no coefficient matching) is the recommended default.

---

## Installation

```bash
pip install -e ".[dev]"
```

**Requirements:** PyTorch ≥ 2.1, transformers, datasets, peft, omegaconf, evaluate, wandb

---

## Project Structure

```
sad-lora/
├── src/
│   ├── spectral_analysis/   # Phase 1–2: teacher SVD, data-weighted subspace, rank selection
│   ├── losses/              # L_align, L_coeff, combined SAD loss, adaptive schedule
│   ├── models/              # SADLoRALayer, SADLoRAModel, spectral initialization
│   ├── training/            # Trainer, DistillationEngine, callbacks
│   ├── evaluation/          # Spectral metrics, task metrics, forgetting metrics
│   └── utils/               # SVD utils, Grassmannian geometry, logging
├── experiments/
│   ├── run_synthetic.py         # Controlled spectral validation (Theorems 1–3)
│   ├── run_roberta_glue.py      # RoBERTa GLUE distillation (main results)
│   ├── run_ablations.py         # Component ablations
│   ├── run_spectral_analysis.py # Alignment score analysis
│   └── run_llama_instruct.py    # LLaMA instruction-following experiment
├── configs/
│   ├── base.yaml                # Shared defaults
│   ├── exp_roberta_glue.yaml    # Main GLUE experiment config
│   └── exp_synthetic.yaml       # Synthetic experiment config
├── scripts/
│   ├── prepare_teacher.sh       # Fine-tune and extract teacher weights
│   ├── extract_teacher_svd.sh   # Compute layerwise spectral targets
│   └── run_all_experiments.sh   # Reproduce all results
└── tests/                       # Unit tests for losses and theorem verification
```

---

## Reproducing Results

### 1. Prepare the teacher

```bash
bash scripts/prepare_teacher.sh          # fine-tune RoBERTa-large on each GLUE task
bash scripts/extract_teacher_svd.sh      # compute data-weighted spectral targets
```

### 2. Run GLUE distillation

```bash
python experiments/run_roberta_glue.py --config configs/exp_roberta_glue.yaml
```

Key config options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_rank` | 8 | LoRA rank r |
| `alpha_align` | 1.0 | Weight of L_align |
| `beta_coeff` | 0.0 | Weight of L_coeff (0 = SAD-LoRA-Align) |
| `kd_temperature` | 4.0 | Distillation temperature |

### 3. Controlled synthetic experiment

```bash
python experiments/run_synthetic.py --config configs/exp_synthetic.yaml
```

Verifies Theorems 1–3 under controlled spectra (sharp / gradual / flat) and covariances.

### 4. Run tests

```bash
pytest tests/ -v
```

---

## Implementation Notes

- **Float32 for spectral ops:** SVD and QR run in float32 even under mixed precision (`torch.amp.autocast("cuda", enabled=False)`).
- **Gradient clipping:** `max_norm=1.0` — SVD gradients spike near degenerate singular values.
- **Buffers not parameters:** `U_target` and `sigma_target` are registered as PyTorch buffers (saved in state_dict, moved with `.to()`, not trained).
- **lora_alpha = r:** SAD-LoRA uses scaling=1.0 because L_coeff explicitly controls adapter magnitude.
- **Implicit SVD:** When `n_calibration < d_in`, compute SVD of `ΔW_T @ X^T / √n` instead of forming the full covariance matrix.

---

## Citation

Coming soon.
