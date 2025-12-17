"""
Simple, reliable Llama fine-tuning script.
Works with environment variables - no complex argument parsing.
Supports: GSM8K dataset OR custom JSONL files.
"""

import os
import json
from typing import List, Dict
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)


def log(msg):
    """Simple logging"""
    print(f"[TRAIN] {msg}")


def load_gsm8k_dataset():
    """Load and format GSM8K dataset"""
    log("Loading GSM8K dataset...")
    ds = load_dataset("gsm8k", "main")
    
    def format_example(ex):
        return {
            "messages": [
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["answer"]},
            ]
        }
    
    # Format as messages
    train_ds = ds["train"].map(lambda ex: format_example(ex))
    eval_ds = ds["test"].map(lambda ex: format_example(ex))
    
    return DatasetDict({"train": train_ds, "validation": eval_ds})


def load_jsonl_dataset(train_path, eval_path):
    """Load custom JSONL dataset"""
    log(f"Loading JSONL: {train_path}")
    
    data_files = {
        "train": train_path,
        "validation": eval_path,
    }
    
    ds = load_dataset("json", data_files=data_files)
    return ds


def messages_to_text(messages: List[Dict]) -> str:
    """Convert messages to training text"""
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "user":
            parts.append(f"Question: {content.strip()}")
        elif role == "assistant":
            parts.append(f"Answer: {content.strip()}")
        else:
            parts.append(content.strip())
    
    return "\\n\\n".join(parts)


def tokenize_dataset(ds, tokenizer, max_length):
    """Tokenize dataset for causal LM"""
    log(f"Tokenizing with max_length={max_length}...")
    
    def tokenize_fn(batch):
        # Handle both "messages" and "prompt"/"completion" formats
        if "messages" in batch:
            texts = [messages_to_text(m) for m in batch["messages"]]
        elif "prompt" in batch and "completion" in batch:
            texts = [f"Question: {p}\\n\\nAnswer: {c}" 
                    for p, c in zip(batch["prompt"], batch["completion"])]
        else:
            raise ValueError("Dataset must have 'messages' or 'prompt'+'completion' fields")
        
        out = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        # Labels are same as input_ids for causal LM
        out["labels"] = out["input_ids"].clone()
        
        return {k: v.tolist() for k, v in out.items()}
    
    tokenized = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=ds["train"].column_names,
        desc="Tokenizing",
    )
    
    return tokenized


def main():
    log("Starting training script")
    log("=" * 70)
    
    # Read configuration from environment
    model_id = os.environ.get("LLM_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    dataset_type = os.environ.get("DATASET_TYPE", "gsm8k")
    train_jsonl = os.environ.get("TRAIN_JSONL", "")
    eval_jsonl = os.environ.get("EVAL_JSONL", "")
    output_dir = os.environ.get("OUTPUT_DIR", "/opt/app-root/src/models/llama-finetuned")
    
    # Training params
    max_steps = int(os.environ.get("MAX_STEPS", "30"))
    num_epochs = int(os.environ.get("NUM_TRAIN_EPOCHS", "1"))
    batch_size = int(os.environ.get("TRAIN_BATCH_SIZE", "1"))
    eval_batch_size = int(os.environ.get("EVAL_BATCH_SIZE", "1"))
    max_seq_length = int(os.environ.get("MAX_SEQ_LENGTH", "512"))
    grad_accum = int(os.environ.get("GRAD_ACCUM_STEPS", "4"))
    lr = float(os.environ.get("LEARNING_RATE", "2e-5"))
    warmup = float(os.environ.get("WARMUP_RATIO", "0.03"))
    save_steps = int(os.environ.get("SAVE_STEPS", "30"))
    eval_steps = int(os.environ.get("EVAL_STEPS", "30"))
    save_limit = int(os.environ.get("SAVE_TOTAL_LIMIT", "2"))
    use_bf16 = os.environ.get("USE_BF16", "1") == "1"
    use_fp16 = os.environ.get("USE_FP16", "0") == "1"
    
    # HF token
    hf_token = os.environ.get("HF_TOKEN")
    
    log(f"Config:")
    log(f"  Model: {model_id}")
    log(f"  Dataset: {dataset_type}")
    log(f"  Output: {output_dir}")
    log(f"  Max steps: {max_steps}")
    log(f"  Batch size: {batch_size}")
    log(f"  Max seq length: {max_seq_length}")
    log("=" * 70)
    
    # Load dataset
    if dataset_type == "gsm8k":
        raw_ds = load_gsm8k_dataset()
    elif dataset_type == "jsonl":
        if not train_jsonl or not eval_jsonl:
            raise ValueError("JSONL dataset requires TRAIN_JSONL and EVAL_JSONL env vars")
        raw_ds = load_jsonl_dataset(train_jsonl, eval_jsonl)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    log(f"Dataset loaded: {len(raw_ds['train'])} train, {len(raw_ds['validation'])} eval")
    
    # Load model and tokenizer
    log(f"Loading model: {model_id}")
    
    # Check if local directory
    is_local = os.path.isdir(model_id)
    
    tokenizer_kwargs = {}
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32,
    }
    
    if is_local:
        log("Using local model (offline mode)")
        tokenizer_kwargs["local_files_only"] = True
        model_kwargs["local_files_only"] = True
    elif hf_token:
        log("Using HF token for authentication")
        tokenizer_kwargs["token"] = hf_token
        model_kwargs["token"] = hf_token
    
    # Load tokenizer with legacy=True to avoid issues
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=False,  # Use slow tokenizer (more stable)
        legacy=True,     # Use legacy behavior
        **tokenizer_kwargs
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    
    log("Tokenizer loaded")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    log("Model loaded")
    
    # Tokenize dataset
    tokenized_ds = tokenize_dataset(raw_ds, tokenizer, max_seq_length)
    log(f"Dataset tokenized")
    
    # Training arguments
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=warmup,
        num_train_epochs=num_epochs,
        max_steps=max_steps if max_steps > 0 else -1,
        logging_steps=10,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_limit,
        gradient_accumulation_steps=grad_accum,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to=["none"],
    )
    
    log("Training arguments configured")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
    )
    
    log("Trainer created")
    log("=" * 70)
    log("Starting training...")
    log("=" * 70)
    
    # Train
    train_result = trainer.train()
    log("Training complete!")
    
    # Evaluate
    log("Running evaluation...")
    metrics = trainer.evaluate()
    log(f"Evaluation metrics: {json.dumps(metrics, indent=2)}")
    
    # Save
    log(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    log("=" * 70)
    log("✅ Training pipeline complete!")
    log("=" * 70)
    
    return metrics


if __name__ == "__main__":
    main()
