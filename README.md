LoRA Fine-Tuning on Tiny GPT-2

This project demonstrates how to fine-tune a small GPT-2 model (sshleifer/tiny-gpt2) using Low-Rank Adaptation (LoRA) for efficient parameter-tuning.
Author: Harshitha Arlapalli
Roll No: 23A91A0573
Project: LLM Fine-Tuning Pipeline using LoRA (Low-Rank Adaptation)

📌 1. Overview

This repository contains a complete end-to-end pipeline for:

Preparing a dataset for supervised fine-tuning

Training a LoRA-augmented causal language model

Evaluating the base vs. fine-tuned model

Exporting LoRA adapters

Uploading the final result to Hugging Face Hub

The model used for this experiment is sshleifer/tiny-gpt2, a very small GPT-2 variant suitable for CPU-only environments and academic demonstration.

📁 2. Repository Structure
llm-lora-finetune/
│── configs/
│   └── config.yaml
│── data/
│   ├── processed/
│   │   ├── train.processed.jsonl
│   │   └── val.processed.jsonl
│── outputs/
│   ├── adapters/          # final LoRA weights
│   ├── evaluation_report.md
│   └── adapters_for_submission.zip
│── src/
│   ├── train.py
│   ├── eval_local.py
│   └── utils.py
│── README.md

🛠️ 3. Environment Setup
Step 1 — Clone repository
git clone <your-repo-url>
cd llm-lora-finetune

Step 2 — Create virtual environment
python -m venv .venv
source .venv/Scripts/activate      # Windows

Step 3 — Install dependencies
pip install -r requirements.txt


All scripts work on Windows + CPU.

📦 4. Dataset Used

The evaluator does not need to regenerate the dataset.
Processed training & validation files are already included:

data/processed/train.processed.jsonl
data/processed/val.processed.jsonl


Each row follows the format:

{
  "instruction": "...",
  "input": "...",
  "output": "..."
}


This allows immediate training without modification.

🚀 5. Running LoRA Fine-Tuning

The full training configuration is defined in:

configs/config.yaml


Run training:

python src/train.py --config configs/config.yaml


This will:

Load sshleifer/tiny-gpt2

Add LoRA layers (rank=8)

Train for 1 epoch

Save adapter weights to:

outputs/adapters/


The evaluator can inspect these weights directly or load them with PEFT.

🔍 6. Running Evaluation

The evaluator can reproduce the evaluation using:

python src/eval_local.py --config configs/config.yaml --max 50


This script:

Loads the base model

Loads the LoRA adapter

Runs inference on 50 evaluation samples

Computes ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum

Saves results into:

outputs/evaluation_report.md


If internet latency causes timeouts (common on Windows), the script automatically retries.

📤 7. LoRA Adapter Export (for submission)

A compressed version of the adapters is provided:

adapters_for_submission.zip


Expanded version exists in:

outputs/adapters/


This directory includes:

adapter_config.json

adapter_model.safetensors

tokenizer adjustments (pad token)

☁️ 8. Hugging Face Model Hub Deployment

Final model repository:

👉 https://huggingface.co/harshithaarlapalli/harshitha-tiny-gpt2-lora

Uploaded contents:

LoRA adapter weights

Model card

Instructions to load adapters

Evaluators can load and run the model as:

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
base = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")

model = PeftModel.from_pretrained(base,
    "harshithaarlapalli/harshitha-tiny-gpt2-lora"
)

out = model.generate(**tok("Hello", return_tensors="pt"), max_new_tokens=20)
print(tok.decode(out[0], skip_special_tokens=True))

📊 9. Expected Evaluation Output

Your evaluation will produce ROUGE scores like:

ROUGE-1: 0.1193
ROUGE-2: 0.1102
ROUGE-L: 0.1193
ROUGE-Lsum: 0.1200


Since the base model is extremely small (2-dim embedding), improvements are minimal — this is normal and expected.

The goal of this assignment is pipeline correctness, not accuracy.
