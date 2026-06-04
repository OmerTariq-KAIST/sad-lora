from setuptools import setup, find_packages

setup(
    name="sad-lora",
    version="0.1.0",
    description="SAD-LoRA: Spectral Alignment Distillation for Low-Rank Adaptation",
    author="SAD-LoRA Authors",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "datasets>=2.16.0",
        "peft>=0.7.0",
        "safetensors>=0.4.0",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "wandb>=0.16.0",
        "omegaconf>=2.3.0",
        "evaluate>=0.4.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "ruff>=0.1.0", "mypy>=1.7.0"],
        "llm": ["vllm>=0.2.0"],
    },
)
