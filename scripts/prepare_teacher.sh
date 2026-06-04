#!/bin/bash
# Fine-tune teacher models on GLUE tasks.
# Uses HuggingFace Trainer for standard full fine-tuning.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

TEACHER_NAME=${1:-"roberta-large"}
OUTPUT_BASE=${2:-"/data/Omer/saved_checkpoints/sad_lora_results/teachers"}
TASKS=${3:-"sst2 mrpc stsb cola qnli rte"}

for TASK in $TASKS; do
    echo "Fine-tuning $TEACHER_NAME on $TASK..."

    python -c "
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import os

task = '$TASK'
model_name = '$TEACHER_NAME'
output_dir = os.path.join('$OUTPUT_BASE', f'{model_name.split(\"/\")[-1]}-{task}')

num_labels = {'sst2': 2, 'mrpc': 2, 'stsb': 1, 'cola': 2, 'qnli': 2, 'rte': 2}[task]
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset('glue', task)
task_keys = {
    'sst2': ('sentence', None), 'mrpc': ('sentence1', 'sentence2'),
    'stsb': ('sentence1', 'sentence2'), 'cola': ('sentence', None),
    'qnli': ('question', 'sentence'), 'rte': ('sentence1', 'sentence2'),
}
k1, k2 = task_keys[task]

def tokenize(examples):
    args = (examples[k1],) if k2 is None else (examples[k1], examples[k2])
    result = tokenizer(*args, truncation=True, max_length=128, padding='max_length')
    result['labels'] = examples['label']
    return result

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset['train'].column_names)

args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    fp16=True,
    report_to='none',
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['validation'],
)
trainer.train()
trainer.save_model(output_dir)
print(f'Teacher saved to {output_dir}')
"
    echo "Done with $TASK"
done

echo "All teachers fine-tuned."
