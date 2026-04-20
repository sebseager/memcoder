"""
Exp 1 — Oracle Ceiling: Train oracle LoRA adapters.

Trains one LoRA adapter per instance via causal-LM next-token prediction on
the pre-patch source file. Uses HuggingFace Trainer for robust mixed-precision
training. Loss-based early stopping with file-size-proportional step cap.

Usage:
    python train_oracle.py --instance-id django__django-12284
    python train_oracle.py --all
    python train_oracle.py --ids-file pilot_ids.txt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset as HFDataset
from peft import PeftModel, get_peft_model
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    LOSS_CURVES_DIR,
    ORACLE_BATCH_SIZE,
    ORACLE_CHUNK_SIZE,
    ORACLE_GRAD_ACCUM,
    ORACLE_LORA_DIR,
    ORACLE_LR,
    ORACLE_MAX_EPOCHS,
    ORACLE_PATIENCE,
    SEED,
)
from helpers import (
    cycle_lora,
    get_file_content_for_instance,
    load_base_model,
    load_subsets,
    load_swebench_dataset,
    load_token_counts,
    make_lora_config,
)


def make_chunk_dataset(
    text: str, tokenizer, chunk_size: int = ORACLE_CHUNK_SIZE
) -> HFDataset:
    """Create a HuggingFace Dataset of non-overlapping token chunks.

    All chunks are padded to chunk_size for uniform batching. Labels are set
    to -100 on padding positions so they don't contribute to the loss.
    """
    pad_id = tokenizer.pad_token_id
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for start in range(0, len(tokens), chunk_size):
        chunk = tokens[start : start + chunk_size]
        if len(chunk) < 64:
            continue
        # Pad to chunk_size
        pad_len = chunk_size - len(chunk)
        labels = chunk.copy() + [-100] * pad_len
        attention_mask = [1] * len(chunk) + [0] * pad_len
        input_ids = chunk + [pad_id] * pad_len
        chunks.append(
            {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }
        )
    return HFDataset.from_list(chunks)


def train_one_oracle(
    instance_id: str,
    file_content: str,
    model,
    tokenizer,
    output_dir: Path,
):
    """Train an oracle LoRA for one instance.

    Returns (metadata_dict, model).  The returned model is a PeftModel;
    pass it back on subsequent calls to avoid adapter stacking.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_chunk_dataset(file_content, tokenizer)
    n_chunks = len(dataset)
    if n_chunks == 0:
        return {
            "instance_id": instance_id,
            "status": "skipped",
            "reason": "no chunks",
        }, model

    # Split 80/20 for early stopping eval set
    if n_chunks >= 5:
        split = dataset.train_test_split(test_size=0.2, seed=SEED)
        train_ds = split["train"]
        eval_ds = split["test"]
    else:
        train_ds = dataset
        eval_ds = dataset

    # Fresh LoRA — wrap on first call, cycle on subsequent
    if isinstance(model, PeftModel):
        model = cycle_lora(model)
    else:
        model = get_peft_model(model, make_lora_config())

    # Scale max steps with file size
    n_tokens = len(tokenizer.encode(file_content, add_special_tokens=False))
    max_steps_cap = max(200, int(n_tokens / 1000 * 100))

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=ORACLE_MAX_EPOCHS,
        max_steps=max_steps_cap,
        per_device_train_batch_size=ORACLE_BATCH_SIZE,
        gradient_accumulation_steps=ORACLE_GRAD_ACCUM,
        learning_rate=ORACLE_LR,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    callbacks = []
    if n_chunks >= 5:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=ORACLE_PATIENCE,
                early_stopping_threshold=0.01,
            )
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    print(
        f"  Training: {len(train_ds)} train, {len(eval_ds)} eval chunks, max {max_steps_cap} steps"
    )

    train_result = trainer.train()

    # Save the final adapter
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Extract loss history
    loss_history = [
        e["eval_loss"] for e in trainer.state.log_history if "eval_loss" in e
    ]

    meta = {
        "instance_id": instance_id,
        "status": "trained",
        "n_chunks": n_chunks,
        "n_train_chunks": len(train_ds),
        "n_tokens": n_tokens,
        "epochs_run": trainer.state.epoch,
        "global_steps": trainer.state.global_step,
        "max_steps_cap": max_steps_cap,
        "final_train_loss": train_result.training_loss,
        "best_eval_loss": trainer.state.best_metric,
        "eval_loss_history": loss_history,
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    del trainer
    torch.cuda.empty_cache()

    return meta, model


def main():
    parser = argparse.ArgumentParser(description="Train oracle LoRAs for Exp 1")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--instance-id", type=str)
    group.add_argument("--ids-file", type=str)
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.instance_id:
        instance_ids = [args.instance_id]
    elif args.ids_file:
        instance_ids = Path(args.ids_file).read_text().strip().split("\n")
    else:
        subsets = load_subsets()
        instance_ids = subsets["constrained_instance_ids"]

    print(f"Training oracle LoRAs for {len(instance_ids)} instances")

    token_counts = load_token_counts()
    swebench_data = load_swebench_dataset()

    print("Loading base model...")
    base_model, tokenizer = load_base_model()

    results = []
    for i, iid in enumerate(instance_ids):
        print(f"\n[{i + 1}/{len(instance_ids)}] {iid}")
        lora_dir = ORACLE_LORA_DIR / iid

        meta_path = lora_dir / "training_meta.json"
        if meta_path.exists():
            print("  Already trained, skipping.")
            with open(meta_path) as f:
                results.append(json.load(f))
            continue

        fpath, content = get_file_content_for_instance(iid, token_counts, swebench_data)
        n_tokens = len(tokenizer.encode(content, add_special_tokens=False))
        print(f"  File: {fpath} ({n_tokens} tokens)")

        t0 = time.time()
        meta, base_model = train_one_oracle(
            iid, content, base_model, tokenizer, lora_dir
        )
        meta["wall_time_s"] = time.time() - t0
        results.append(meta)

        LOSS_CURVES_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOSS_CURVES_DIR / f"{iid}.json", "w") as f:
            json.dump(meta.get("eval_loss_history", []), f)

        with open(lora_dir / "training_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    ORACLE_LORA_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = ORACLE_LORA_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTraining summary saved to {summary_path}")

    trained = [r for r in results if r.get("status") == "trained"]
    if trained:
        losses = [r["final_train_loss"] for r in trained]
        print(
            f"  Trained: {len(trained)}, mean final train loss: {np.mean(losses):.4f}"
        )


if __name__ == "__main__":
    main()
