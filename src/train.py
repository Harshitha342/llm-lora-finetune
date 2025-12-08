# src/train.py
import os
import yaml
import logging
import random
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_bnb_config(cfg):
    quant = cfg.get("quant", {})
    if not quant.get("enabled", False):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=quant.get("load_in_4bit", True),
        bnb_4bit_use_double_quant=quant.get("double_quant", True),
        bnb_4bit_quant_type=quant.get("quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch.float16,
    )

def get_tokenizer_and_model(cfg, bnb_config=None):
    model_name = cfg["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=bnb_config if bnb_config is not None else None,
        trust_remote_code=False,
    )
    return tokenizer, model

def prepare_peft_model(model, cfg):
    lcfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lcfg.get("r", 8),
        lora_alpha=lcfg.get("alpha", 16),
        target_modules=lcfg.get("target_modules", None),
        lora_dropout=lcfg.get("dropout", 0.05),
        bias=lcfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    return model

def load_processed_dataset(cfg):
    processed_dir = Path(cfg["dataset"]["processed_dir"])
    files = {
        "train": str(processed_dir / "train.processed.jsonl"),
        "validation": str(processed_dir / "val.processed.jsonl"),
    }
    ds = load_dataset("json", data_files=files)
    return ds

def tokenize_function(examples, tokenizer, max_length):
    texts = examples.get("text")
    return tokenizer(texts, truncation=True, padding="longest", max_length=max_length)

def main(config_path="configs/config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    bnb_config = build_bnb_config(cfg)

    tokenizer, model = get_tokenizer_and_model(cfg, bnb_config=bnb_config)
    model = prepare_peft_model(model, cfg)

    ds = load_processed_dataset(cfg)
    max_length = cfg.get("max_length", 2048)
    tokenized = ds.map(lambda ex: tokenize_function(ex, tokenizer, max_length),
                       batched=True, remove_columns=ds["train"].column_names)

    tr_cfg = cfg["training"]
    report_to = ["wandb"] if cfg.get("monitoring", {}).get("use_wandb", False) else ["none"]

    training_args = TrainingArguments(
        output_dir=tr_cfg["output_dir"],
        per_device_train_batch_size=tr_cfg.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=tr_cfg.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=tr_cfg.get("gradient_accumulation_steps", 1),
        num_train_epochs=tr_cfg.get("num_train_epochs", 3),
        learning_rate=tr_cfg.get("learning_rate", 2e-4),
        weight_decay=tr_cfg.get("weight_decay", 0.0),
        logging_steps=tr_cfg.get("logging_steps", 50),
        save_steps=tr_cfg.get("save_steps", 500),
        evaluation_strategy="steps",
        eval_steps=tr_cfg.get("eval_steps", 200),
        save_total_limit=3,
        fp16=True if torch.cuda.is_available() else False,
        report_to=report_to,
        run_name=cfg.get("experiment_name", "llm-lora-run"),
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
    )

    logger.info("Starting training")
    trainer.train()

    adapter_dir = Path(tr_cfg["adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    logger.info(f"Saved LoRA adapter to {adapter_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
