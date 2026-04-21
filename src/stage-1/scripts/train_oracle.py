from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from config import (
    MODEL_ID,
    ORACLE_CHUNK_SIZE,
    ORACLE_GRAD_ACCUM,
    ORACLE_LORA_DIR,
    ORACLE_LR,
    ORACLE_MAX_EPOCHS,
    ORACLE_MIN_EPOCHS,
    ORACLE_PATIENCE,
)
from datasets import Dataset as HFDataset
from helpers import (
    build_file_records,
    cycle_lora_adapter,
    load_instances,
    load_model_and_tokenizer,
    make_chunk_dataset_records,
    make_lora_config,
    prepare_model_for_oracle_training,
    select_file_records,
    set_seed,
)
from peft import PeftModel, get_peft_model
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


class MinEpochEarlyStopping(EarlyStoppingCallback):
    def __init__(self, min_epochs: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_epochs = min_epochs

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.epoch is not None and state.epoch < self.min_epochs:
            return control
        return super().on_evaluate(args, state, control, metrics=metrics, **kwargs)


def filter_file_records(records: list, include_keys: set[str] | None) -> list:
    if not include_keys:
        return records
    return [r for r in records if r.file_key in include_keys]


def train_one_file(
    model,
    tokenizer,
    file_key: str,
    file_text: str,
    out_dir: Path,
    chunk_size: int,
    lr: float,
    min_epochs: int,
    max_epochs: int,
    grad_accum: int,
    batch_size: int,
    patience: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    records = make_chunk_dataset_records(file_text, tokenizer, chunk_size=chunk_size)
    if not records:
        meta = {
            "file_key": file_key,
            "status": "skipped",
            "reason": "no_valid_chunks",
        }
        return meta, model

    ds = HFDataset.from_list(records)
    if len(ds) >= 5:
        split = ds.train_test_split(test_size=0.2, seed=42)
        train_ds = split["train"]
        eval_ds = split["test"]
    else:
        train_ds = ds
        eval_ds = ds

    if isinstance(model, PeftModel):
        model = cycle_lora_adapter(model)
    else:
        model = prepare_model_for_oracle_training(model)
        model = get_peft_model(model, make_lora_config())

    n_tokens = len(tokenizer.encode(file_text, add_special_tokens=False))

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=max_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=lr,
        bf16=torch.cuda.is_available(),
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
        report_to="none",
        remove_unused_columns=False,
        save_total_limit=1,
        dataloader_pin_memory=False,
    )

    callbacks = []
    if len(ds) >= 5:
        callbacks.append(
            MinEpochEarlyStopping(
                min_epochs=min_epochs,
                early_stopping_patience=patience,
                early_stopping_threshold=0.001,
            )
        )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    result = trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    eval_losses = [
        x["eval_loss"] for x in trainer.state.log_history if "eval_loss" in x
    ]
    meta = {
        "file_key": file_key,
        "status": "trained",
        "n_tokens": n_tokens,
        "n_chunks": len(ds),
        "n_train_chunks": len(train_ds),
        "n_eval_chunks": len(eval_ds),
        "global_step": trainer.state.global_step,
        "epochs": trainer.state.epoch,
        "final_train_loss": result.training_loss,
        "best_eval_loss": trainer.state.best_metric,
        "eval_loss_history": eval_losses,
    }

    with (out_dir / "training_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return meta, model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage 1 oracle LoRA adapters")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--file-keys-file", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--chunk-size", type=int, default=ORACLE_CHUNK_SIZE)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=ORACLE_GRAD_ACCUM)
    p.add_argument("--lr", type=float, default=ORACLE_LR)
    p.add_argument("--min-epochs", type=int, default=ORACLE_MIN_EPOCHS)
    p.add_argument("--max-epochs", type=int, default=ORACLE_MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=ORACLE_PATIENCE)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed()

    instances = load_instances()
    file_records = build_file_records(instances)

    include_keys = None
    if args.file_keys_file:
        include_keys = {
            line.strip()
            for line in Path(args.file_keys_file)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        }
        file_records = filter_file_records(file_records, include_keys)

    file_records = select_file_records(file_records, args.max_files)

    ORACLE_LORA_DIR.mkdir(parents=True, exist_ok=True)

    index = [
        {
            "file_key": fr.file_key,
            "repo": fr.repo,
            "file_path": fr.file_path,
        }
        for fr in file_records
    ]
    with (ORACLE_LORA_DIR / "file_index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"Target file adapters: {len(file_records)}")
    if args.dry_run:
        return 0

    model, tokenizer = load_model_and_tokenizer(model_id=args.model_id, use_4bit=True)

    rows = []
    for i, fr in enumerate(file_records):
        print(f"[{i + 1}/{len(file_records)}] {fr.file_key}")
        out_dir = ORACLE_LORA_DIR / fr.file_key
        meta_path = out_dir / "training_meta.json"
        if meta_path.exists() and not args.force:
            print("  skip existing adapter")
            with meta_path.open("r", encoding="utf-8") as f:
                rows.append(json.load(f))
            continue

        t0 = time.time()
        meta, model = train_one_file(
            model=model,
            tokenizer=tokenizer,
            file_key=fr.file_key,
            file_text=fr.full_file,
            out_dir=out_dir,
            chunk_size=args.chunk_size,
            lr=args.lr,
            min_epochs=args.min_epochs,
            max_epochs=args.max_epochs,
            grad_accum=args.grad_accum,
            batch_size=args.batch_size,
            patience=args.patience,
        )
        meta["wall_time_s"] = time.time() - t0
        rows.append(meta)

    with (ORACLE_LORA_DIR / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    trained = [r for r in rows if r.get("status") == "trained"]
    if trained:
        losses = [r["final_train_loss"] for r in trained]
        print(f"Trained {len(trained)} adapters, mean train loss={np.mean(losses):.4f}")
    else:
        print("No adapters trained in this run")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
