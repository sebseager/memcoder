from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from config import (
    MODEL_ID,
    ORACLE_BEHAVIORAL_EPOCHS,
    ORACLE_BEHAVIORAL_LR_MULT,
    ORACLE_BEHAVIORAL_PROBES,
    ORACLE_CHUNK_SIZE,
    ORACLE_GRAD_ACCUM,
    ORACLE_LR,
    ORACLE_MAX_EPOCHS,
    ORACLE_MIN_EPOCHS,
    ORACLE_PATIENCE,
    SEED,
    TRUNCATION_BUDGET_TOKENS,
    get_stage1_paths,
)
from datasets import Dataset as HFDataset
from helpers import (
    build_behavioral_probe_manifest,
    build_file_records,
    build_function_examples_from_file,
    build_function_examples_from_instances,
    build_supervised_records_from_examples,
    cycle_lora_adapter,
    inspect_first_supervised_example,
    load_instances,
    load_json,
    load_model_and_tokenizer,
    make_file_key,
    make_lora_config,
    prepare_model_for_oracle_training,
    select_file_records,
    set_seed,
    write_json,
)
from peft import PeftModel, get_peft_model
from transformers import (
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


class SupervisedCausalCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            n = len(f["input_ids"])
            pad_n = max_len - n
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_n)
            attention_mask.append(f["attention_mask"] + [0] * pad_n)
            labels.append(f["labels"] + [-100] * pad_n)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def filter_file_records(records: list, include_keys: set[str] | None) -> list:
    if not include_keys:
        return records
    return [r for r in records if r.file_key in include_keys]


def _float_close(a, b, atol: float = 1e-12) -> bool:
    try:
        return abs(float(a) - float(b)) <= atol
    except (TypeError, ValueError):
        return False


def build_instances_by_file_key(instances: list[dict]) -> dict[str, list[dict]]:
    by_key: dict[str, list[dict]] = {}
    for row in instances:
        key = make_file_key(row["repo"], row["file_path"])
        by_key.setdefault(key, []).append(row)
    return by_key


def build_examples_for_file(
    *,
    file_text: str,
    instances_for_file: list[dict],
    tokenizer,
    trunc_budget: int,
) -> list:
    examples = build_function_examples_from_file(
        file_text=file_text,
        tokenizer=tokenizer,
        trunc_budget_tokens=trunc_budget,
    )
    if examples:
        return examples
    return build_function_examples_from_instances(
        instances_for_file=instances_for_file,
        tokenizer=tokenizer,
        trunc_budget_tokens=trunc_budget,
    )


def existing_meta_compatible(
    meta: dict,
    args: argparse.Namespace,
    probe_manifest_sha256: str,
    *,
    behavioral_probes: int,
    behavioral_epochs: int,
    behavioral_lr_mult: float,
) -> tuple[bool, str]:
    if meta.get("objective") != "supervised_prompt_completion_masked":
        return False, "objective_mismatch"

    if meta.get("status") != "trained":
        return False, "status_not_trained"

    checks = {
        "model_id": args.model_id,
        "chunk_size": args.chunk_size,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "min_epochs": args.min_epochs,
        "max_epochs": args.max_epochs,
        "trunc_budget": args.trunc_budget,
        "enable_eval": bool(args.enable_eval),
        "behavioral_probes": behavioral_probes,
        "behavioral_epochs": behavioral_epochs,
        "seed": args.seed,
        "behavioral_probe_manifest_sha256": probe_manifest_sha256,
    }

    for key, expected in checks.items():
        got = meta.get(key)
        if got != expected:
            return False, f"{key}_mismatch"

    if not _float_close(meta.get("lr"), args.lr):
        return False, "lr_mismatch"

    if not _float_close(meta.get("behavioral_lr_mult", 1.0), behavioral_lr_mult):
        return False, "behavioral_lr_mult_mismatch"

    return True, "compatible"


def train_one_file(
    model,
    tokenizer,
    model_id: str,
    file_key: str,
    file_text: str,
    examples: list,
    probe_manifest: dict,
    out_dir: Path,
    chunk_size: int,
    trunc_budget: int,
    lr: float,
    min_epochs: int,
    max_epochs: int,
    grad_accum: int,
    batch_size: int,
    patience: int,
    enable_eval: bool,
    behavioral_probes: int,
    behavioral_epochs: int,
    behavioral_lr_mult: float,
    seed: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    source = examples[0].source if examples else "none"

    records, supervised_stats = build_supervised_records_from_examples(
        examples=examples,
        tokenizer=tokenizer,
        max_sequence_tokens=chunk_size,
    )

    if not records:
        meta = {
            "file_key": file_key,
            "status": "skipped",
            "reason": "no_supervised_records",
            "model_id": model_id,
            "objective": "supervised_prompt_completion_masked",
            "chunk_size": chunk_size,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "lr": lr,
            "min_epochs": min_epochs,
            "max_epochs": max_epochs,
            "trunc_budget": trunc_budget,
            "behavioral_probes": behavioral_probes,
            "behavioral_epochs": behavioral_epochs,
            "behavioral_lr_mult": behavioral_lr_mult,
            "behavioral_probe_manifest_sha256": probe_manifest.get("manifest_sha256"),
            "seed": seed,
        }
        return meta, model

    ds = HFDataset.from_list(records)
    if enable_eval and len(ds) >= 5:
        split = ds.train_test_split(test_size=0.2, seed=seed)
        train_ds = split["train"]
        eval_ds = split["test"]
    else:
        train_ds = ds
        eval_ds = ds if enable_eval else None

    if isinstance(model, PeftModel):
        model = cycle_lora_adapter(model)
    else:
        model = prepare_model_for_oracle_training(model)
        model = get_peft_model(model, make_lora_config())

    n_tokens = len(tokenizer.encode(file_text, add_special_tokens=False))

    eval_strategy = "epoch" if enable_eval else "no"
    save_strategy = "epoch" if enable_eval else "no"
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
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=enable_eval,
        metric_for_best_model="eval_loss" if enable_eval else None,
        greater_is_better=False if enable_eval else None,
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
        save_total_limit=1,
        dataloader_pin_memory=False,
    )

    callbacks = []
    if enable_eval and len(ds) >= 5:
        callbacks.append(
            MinEpochEarlyStopping(
                min_epochs=min_epochs,
                early_stopping_patience=patience,
                early_stopping_threshold=0.001,
            )
        )

    collator = SupervisedCausalCollator(pad_token_id=tokenizer.pad_token_id)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    result = trainer.train()

    behavioral_status = "disabled"
    behavioral_records = 0
    behavioral_truncated_records = 0
    behavioral_final_train_loss = None
    behavioral_train_loss_history = []
    if behavioral_probes > 0:
        probe_examples = examples[:behavioral_probes]
        probe_records, probe_stats = build_supervised_records_from_examples(
            examples=probe_examples,
            tokenizer=tokenizer,
            max_sequence_tokens=chunk_size,
        )
        behavioral_records = len(probe_records)
        behavioral_truncated_records = int(probe_stats["n_truncated"])

        if probe_records:
            probe_ds = HFDataset.from_list(probe_records)
            probe_args = TrainingArguments(
                output_dir=str(out_dir / "behavioral_checkpoints"),
                num_train_epochs=behavioral_epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=1,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                learning_rate=lr * behavioral_lr_mult,
                bf16=torch.cuda.is_available(),
                logging_steps=1,
                eval_strategy="no",
                save_strategy="no",
                seed=seed,
                report_to="none",
                remove_unused_columns=False,
                dataloader_pin_memory=False,
            )
            probe_trainer = Trainer(
                model=model,
                args=probe_args,
                train_dataset=probe_ds,
                data_collator=collator,
            )
            probe_result = probe_trainer.train()
            behavioral_status = "trained"
            behavioral_final_train_loss = float(probe_result.training_loss)
            behavioral_train_loss_history = [
                x["loss"] for x in probe_trainer.state.log_history if "loss" in x
            ]
            del probe_trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            behavioral_status = "no_records"

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    eval_losses = [
        x["eval_loss"] for x in trainer.state.log_history if "eval_loss" in x
    ]
    train_losses = [x["loss"] for x in trainer.state.log_history if "loss" in x]
    meta = {
        "file_key": file_key,
        "status": "trained",
        "model_id": model_id,
        "objective": "supervised_prompt_completion_masked",
        "chunk_size": chunk_size,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lr": lr,
        "min_epochs": min_epochs,
        "max_epochs": max_epochs,
        "trunc_budget": trunc_budget,
        "enable_eval": bool(enable_eval),
        "behavioral_probes": behavioral_probes,
        "behavioral_epochs": behavioral_epochs,
        "behavioral_lr_mult": behavioral_lr_mult,
        "behavioral_probe_manifest_sha256": probe_manifest.get("manifest_sha256"),
        "behavioral_status": behavioral_status,
        "behavioral_records": behavioral_records,
        "behavioral_truncated_records": behavioral_truncated_records,
        "behavioral_final_train_loss": behavioral_final_train_loss,
        "behavioral_train_loss_history": behavioral_train_loss_history,
        "n_tokens": n_tokens,
        "n_examples": supervised_stats["n_examples"],
        "n_records": supervised_stats["n_records"],
        "n_truncated_records": supervised_stats["n_truncated"],
        "mean_prompt_tokens": supervised_stats["mean_prompt_tokens"],
        "mean_target_tokens": supervised_stats["mean_target_tokens"],
        "example_source": source,
        "seed": seed,
        "n_train_records": len(train_ds),
        "n_eval_records": 0 if eval_ds is None else len(eval_ds),
        "global_step": trainer.state.global_step,
        "epochs": trainer.state.epoch,
        "final_train_loss": result.training_loss,
        "train_loss_history": train_losses,
        "best_eval_loss": trainer.state.best_metric,
        "eval_loss_history": eval_losses,
    }

    with (out_dir / "training_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    adapter_meta = {
        "file_key": file_key,
        "model_id": model_id,
        "objective": "supervised_prompt_completion_masked",
        "behavioral_probe_manifest": probe_manifest,
    }
    write_json(out_dir / "adapter_metadata.json", adapter_meta)

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
    p.add_argument(
        "--chunk-size",
        type=int,
        default=ORACLE_CHUNK_SIZE,
        help="Maximum supervised sequence length in tokens.",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=ORACLE_GRAD_ACCUM)
    p.add_argument("--lr", type=float, default=ORACLE_LR)
    p.add_argument("--min-epochs", type=int, default=ORACLE_MIN_EPOCHS)
    p.add_argument("--max-epochs", type=int, default=ORACLE_MAX_EPOCHS)
    p.add_argument("--patience", type=int, default=ORACLE_PATIENCE)
    p.add_argument("--trunc-budget", type=int, default=TRUNCATION_BUDGET_TOKENS)
    p.add_argument(
        "--enable-eval",
        action="store_true",
        help="Enable per-epoch eval/early-stopping. Disabled by default to reduce VRAM pressure.",
    )
    p.add_argument(
        "--inspect-only-file-key",
        default=None,
        help="Build one supervised example for this file key and print inspection JSON.",
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def assert_saved_probe_manifest_compatible(
    adapter_dir: Path,
    current_probe_manifest: dict,
) -> None:
    adapter_meta = load_json(adapter_dir / "adapter_metadata.json")
    if not isinstance(adapter_meta, dict):
        return

    saved_manifest = adapter_meta.get("behavioral_probe_manifest")
    if not isinstance(saved_manifest, dict):
        return

    saved_sha = saved_manifest.get("manifest_sha256")
    current_sha = current_probe_manifest.get("manifest_sha256")
    if saved_sha and current_sha and saved_sha != current_sha:
        raise RuntimeError(
            "Refusing retrain due to behavioral probe manifest mismatch "
            f"for adapter {adapter_dir.name}. Delete the adapter directory intentionally "
            "before retraining to avoid mixing probe-generation logic."
        )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    paths = get_stage1_paths(args.model_id)
    behavioral_probes = ORACLE_BEHAVIORAL_PROBES
    behavioral_epochs = ORACLE_BEHAVIORAL_EPOCHS
    behavioral_lr_mult = ORACLE_BEHAVIORAL_LR_MULT

    instances = load_instances()
    file_records = build_file_records(instances)
    instances_by_key = build_instances_by_file_key(instances)

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

    paths.oracle_lora.mkdir(parents=True, exist_ok=True)

    index = [
        {
            "file_key": fr.file_key,
            "repo": fr.repo,
            "file_path": fr.file_path,
        }
        for fr in file_records
    ]
    with (paths.oracle_lora / "file_index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"Target file adapters: {len(file_records)}")
    if args.dry_run:
        return 0

    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id,
        use_4bit=True,
        seed=args.seed,
    )

    if args.inspect_only_file_key:
        target = next(
            (fr for fr in file_records if fr.file_key == args.inspect_only_file_key),
            None,
        )
        if target is None:
            print(f"Unknown file key: {args.inspect_only_file_key}")
            return 1

        examples = build_examples_for_file(
            file_text=target.full_file,
            instances_for_file=instances_by_key.get(target.file_key, []),
            tokenizer=tokenizer,
            trunc_budget=args.trunc_budget,
        )

        inspection = inspect_first_supervised_example(
            examples=examples,
            tokenizer=tokenizer,
            max_sequence_tokens=args.chunk_size,
        )
        if inspection is None:
            print("No supervised example available for inspection")
            return 1

        print(json.dumps(inspection, indent=2))
        return 0

    rows = []
    for i, fr in enumerate(file_records):
        print(f"[{i + 1}/{len(file_records)}] {fr.file_key}")
        out_dir = paths.oracle_lora / fr.file_key
        examples = build_examples_for_file(
            file_text=fr.full_file,
            instances_for_file=instances_by_key.get(fr.file_key, []),
            tokenizer=tokenizer,
            trunc_budget=args.trunc_budget,
        )
        probe_manifest = build_behavioral_probe_manifest(
            examples=examples,
            n_probes=behavioral_probes,
        )
        assert_saved_probe_manifest_compatible(out_dir, probe_manifest)

        meta_path = out_dir / "training_meta.json"
        if meta_path.exists() and not args.force:
            with meta_path.open("r", encoding="utf-8") as f:
                existing_meta = json.load(f)

            compatible, reason = existing_meta_compatible(
                existing_meta,
                args,
                probe_manifest_sha256=probe_manifest.get("manifest_sha256", ""),
                behavioral_probes=behavioral_probes,
                behavioral_epochs=behavioral_epochs,
                behavioral_lr_mult=behavioral_lr_mult,
            )
            if compatible:
                print("  skip existing compatible adapter")
                rows.append(existing_meta)
                continue

            print(f"  existing adapter incompatible ({reason}); retraining")
            shutil.rmtree(out_dir, ignore_errors=True)

        t0 = time.time()
        meta, model = train_one_file(
            model=model,
            tokenizer=tokenizer,
            model_id=args.model_id,
            file_key=fr.file_key,
            file_text=fr.full_file,
            examples=examples,
            probe_manifest=probe_manifest,
            out_dir=out_dir,
            chunk_size=args.chunk_size,
            trunc_budget=args.trunc_budget,
            lr=args.lr,
            min_epochs=args.min_epochs,
            max_epochs=args.max_epochs,
            grad_accum=args.grad_accum,
            batch_size=args.batch_size,
            patience=args.patience,
            enable_eval=args.enable_eval,
            behavioral_probes=behavioral_probes,
            behavioral_epochs=behavioral_epochs,
            behavioral_lr_mult=behavioral_lr_mult,
            seed=args.seed,
        )
        meta["wall_time_s"] = time.time() - t0
        rows.append(meta)

    with (paths.oracle_lora / "training_summary.json").open("w", encoding="utf-8") as f:
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
