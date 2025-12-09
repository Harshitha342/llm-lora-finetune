LoRA Fine-Tuning on Tiny GPT-2

This project demonstrates how to fine-tune a small GPT-2 model (sshleifer/tiny-gpt2) using Low-Rank Adaptation (LoRA) for efficient parameter-tuning.
The final LoRA adapter is uploaded on HuggingFace:

🔗 HuggingFace Model:
https://huggingface.co/harshithaarlapalli/harshitha-tiny-gpt2-lora

📌 Project Structure
llm-lora-finetune/
│
├── configs/
│   └── config.yaml          # Training configuration
│
├── src/
│   ├── train.py             # LoRA fine-tuning script
│   ├── eval.py              # Evaluation script (HuggingFace datasets)
│   └── eval_local.py        # Local evaluation (offline-safe)
│
├── outputs/
│   └── adapters/            # (Ignored in GitHub) LoRA adapter weights
│
└── adapters_for_submission.zip   # Uploaded to HuggingFace

🧠 What This Project Does

Loads a base GPT-2 tiny model

Injects LoRA layers on attention modules

Fine-tunes on custom instruction–response training data

Saves adapter weights efficiently (<2MB)

Tests model with both base and LoRA-enabled generations

Computes ROUGE scores for comparison

🚀 Training

Run:

python src/train.py --config configs/config.yaml


This will:

Load the dataset

Apply LoRA configuration

Train for 1 epoch

Save the adapter to outputs/adapters/

📝 Evaluation
Online / HF-dataset-based:
python src/eval.py --config configs/config.yaml

Offline / local evaluation:
python src/eval_local.py --config configs/config.yaml --max 50

📦 Export for Submission

A zipped folder containing the LoRA adapter:

adapters_for_submission.zip


Uploaded to HuggingFace model hub.

