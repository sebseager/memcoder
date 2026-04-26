#!/usr/bin/env python3
"""Run the initial MemCoder SHINE evaluation from artifact files.

This script adapts the flow from `vendor/SHINE/inference.ipynb` into a CLI:

- load Qwen + the SHINE metanetwork checkpoint
- generate a LoRA dictionary from one design document
- run generated QA pairs under three conditions:
  - `naive`: question only
  - `in_context`: design document in the prompt
  - `shine`: generated LoRA loaded, no document in the prompt
- write one JSONL record per question/condition
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("memcoder.shine_eval")
torch = None
OmegaConf = None
AutoTokenizer = None


def import_runtime_deps() -> None:
    global AutoTokenizer, OmegaConf, torch
    import torch as torch_module
    from omegaconf import OmegaConf as omega_conf
    from transformers import AutoTokenizer as auto_tokenizer

    torch = torch_module
    OmegaConf = omega_conf
    AutoTokenizer = auto_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=Path("config/shine_eval_demo.yaml"),
        type=Path,
        help="MemCoder SHINE eval config file.",
    )
    parser.add_argument("--design-doc", type=Path)
    parser.add_argument("--qa-pairs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shine-root", type=Path)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Checkpoint directory containing metanetwork.pth and metalora.pth.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Override cfg.paths.model_path / cfg.model.model_from / tokenizer_from.",
    )
    parser.add_argument(
        "--save-lora-dict",
        type=Path,
        help="Optional path to save the generated LoRA dictionary with torch.save.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        choices=["naive", "in_context", "shine"],
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--context-max-length", type=int)
    parser.add_argument("--conversation-max-length", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--allow-random-metanetwork",
        action="store_true",
        help="Allow SHINE condition to run without a checkpoint. Intended only for smoke tests.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="OmegaConf dotlist override, e.g. --set model.lora_r=8.",
    )
    return parser.parse_args()


def load_json_or_text(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return path.read_text(encoding="utf-8")


def load_design_doc(path: Path) -> tuple[str, dict[str, Any]]:
    payload = load_json_or_text(path)
    if isinstance(payload, str):
        return payload, {"document_path": str(path)}
    if not isinstance(payload, dict):
        raise ValueError(f"Design doc must be text or a JSON object: {path}")

    for key in ("document", "source_document_text", "text", "context"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value, payload

    raise ValueError(
        f"Could not find design doc text in {path}; expected one of "
        "`document`, `source_document_text`, `text`, or `context`."
    )


def _answer_from_record(record: dict[str, Any]) -> Any:
    for key in ("answer", "ground_truth", "expected_answer", "reference_answer"):
        if key in record:
            return record[key]
    return None


def load_qa_pairs(path: Path) -> list[dict[str, Any]]:
    payload = load_json_or_text(path)
    if isinstance(payload, dict):
        if "qa_pairs" in payload:
            records = payload["qa_pairs"]
        elif "questions" in payload:
            records = payload["questions"]
        else:
            raise ValueError(f"Could not find `qa_pairs` or `questions` in {path}")
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"QA pairs must be JSON or JSONL: {path}")

    normalized = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"QA record {idx} is not an object")
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"QA record {idx} is missing a non-empty `question`")
        normalized.append(
            {
                "qa_id": record.get("qa_id") or record.get("id") or f"qa_{idx + 1:04d}",
                "question": question,
                "expected_answer": _answer_from_record(record),
                "metadata": record,
            }
        )
    return normalized


def setup_shine_imports(shine_root: Path) -> None:
    shine_root = shine_root.resolve()
    if not shine_root.exists():
        raise FileNotFoundError(f"SHINE root does not exist: {shine_root}")
    sys.path.insert(0, str(shine_root))


def load_config(args: argparse.Namespace):
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    eval_cfg = cfg.get("memcoder_eval", {})

    def cfg_path(name: str) -> Path | None:
        value = eval_cfg.get(name)
        return Path(str(value)) if value is not None else None

    args.design_doc = args.design_doc or cfg_path("design_doc")
    args.qa_pairs = args.qa_pairs or cfg_path("qa_pairs")
    args.output = args.output or cfg_path("output")
    args.shine_root = args.shine_root or cfg_path("shine_root") or Path("vendor/SHINE")
    args.checkpoint_dir = args.checkpoint_dir or cfg_path("checkpoint_dir")
    args.model_path = args.model_path or cfg_path("model_path")
    args.save_lora_dict = args.save_lora_dict or cfg_path("save_lora_dict")
    args.conditions = args.conditions or list(
        eval_cfg.get("conditions", ["naive", "in_context", "shine"])
    )
    args.device = args.device or str(eval_cfg.get("device", cfg.run.get("device", "auto")))
    args.context_max_length = args.context_max_length or int(
        eval_cfg.get("context_max_length", cfg.test.context_max_length)
    )
    args.conversation_max_length = args.conversation_max_length or int(
        eval_cfg.get("conversation_max_length", cfg.test.conversation_max_length)
    )
    args.max_new_tokens = args.max_new_tokens or int(
        eval_cfg.get("max_new_tokens", cfg.test.max_new_tokens)
    )
    args.seed = args.seed or int(eval_cfg.get("seed", cfg.run.get("seed", 42)))

    missing = [
        name
        for name in ("design_doc", "qa_pairs", "output")
        if getattr(args, name) is None
    ]
    if missing:
        missing_flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(
            f"Missing required eval paths: {missing_flags}. Provide them on the CLI "
            f"or in {args.config} under `memcoder_eval`."
        )

    if args.model_path is not None:
        model_path = str(args.model_path)
        cfg.paths.model_path = model_path
        cfg.model.model_from = model_path
        cfg.model.tokenizer_from = model_path

    cfg.run.seed = args.seed
    cfg.test.context_max_length = args.context_max_length
    cfg.test.conversation_max_length = args.conversation_max_length
    cfg.test.max_new_tokens = args.max_new_tokens
    return cfg


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_think_and_answer(text: str) -> tuple[str, str]:
    think = ""
    answer = text.strip()
    lower = text.lower()
    start_tag = "<think>"
    end_tag = "</think>"
    start = lower.find(start_tag)
    end = lower.find(end_tag)

    if start != -1 and end != -1 and end > start:
        think = text[start + len(start_tag) : end].strip()
        answer = text[end + len(end_tag) :].strip()
    else:
        answer = re.sub(
            r"<think>.*?</think>\s*", "", text, flags=re.IGNORECASE | re.DOTALL
        ).strip()

    answer = re.sub(r"^(final answer|answer)\s*:\s*", "", answer, flags=re.IGNORECASE).strip()
    return think, answer


def load_shine_runtime(cfg: Any, args: argparse.Namespace, device: torch.device):
    from metanetwork_family import Metanetwork
    from utils.myfreeze import freeze
    from utils.myinit import _import_class
    from utils.mysaveload import load_checkpoint
    from utils.myseed import set_seed

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(int(cfg.run.seed))

    meta_model_cls = _import_class(cfg.model.metamodel_class_path)
    config_cls = _import_class(cfg.model.config_class_path)

    LOGGER.info("Loading model config from %s", cfg.model.model_from)
    config = config_cls.from_pretrained(cfg.model.model_from)
    config.num_mem_token = -1
    cfg.hidden_size = config.hidden_size
    cfg.num_layers = config.num_hidden_layers

    LOGGER.info("Calculating SHINE memory token count")
    with torch.device("meta"):
        tmp_model = meta_model_cls(config)
    lora_params = tmp_model.lora_params_numel(cfg.model.lora_r)
    base_params = cfg.hidden_size * cfg.num_layers
    if lora_params % base_params != 0:
        raise ValueError(
            f"lora_params ({lora_params}) must be divisible by hidden*layers ({base_params})"
        )
    config.num_mem_token = lora_params // base_params
    cfg.num_mem_token = config.num_mem_token
    del tmp_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    LOGGER.info("Loading tokenizer from %s", cfg.model.tokenizer_from)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_from, padding_side="left", use_fast=True
    )
    tokenizer.add_tokens(["<RECON>", "<COMP>", "<NOTHING>"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Loading Qwen model from %s", cfg.model.model_from)
    metamodel = meta_model_cls.from_pretrained(cfg.model.model_from, config=config)
    metamodel.reset_mem_tokens()
    metamodel.resize_token_embeddings(len(tokenizer))

    LOGGER.info("Initializing SHINE metanetwork")
    metanetwork = Metanetwork(metamodel, cfg, metamodel.lora_params_numel(cfg.model.lora_r))
    metanetwork.to(device)
    freeze(metamodel)

    metalora = None
    if args.checkpoint_dir is not None:
        LOGGER.info("Loading SHINE checkpoint from %s", args.checkpoint_dir)
        metanetwork, metalora, _ = load_checkpoint(metanetwork, str(args.checkpoint_dir), device)
    elif "shine" in args.conditions and not args.allow_random_metanetwork:
        raise ValueError(
            "`shine` condition requires --checkpoint-dir unless "
            "--allow-random-metanetwork is set."
        )
    else:
        LOGGER.warning("No checkpoint supplied; using initialized metanetwork weights")

    metanetwork.eval()
    return metanetwork, tokenizer, metalora


def build_lora_dict(
    metanetwork: Any,
    tokenizer: Any,
    document: str,
    metalora: Any,
    device: torch.device,
    context_max_length: int,
) -> Any:
    with torch.no_grad():
        enc = tokenizer(
            [document],
            max_length=context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        evidence_ids = enc["input_ids"].to(device)
        evidence_attention_mask = enc["attention_mask"].to(device)
        return metanetwork.generate_lora_dict(evidence_ids, evidence_attention_mask, metalora)


def messages_for_condition(condition: str, document: str, question: str) -> list[dict[str, str]]:
    if condition == "in_context":
        prompt = (
            "You are a helpful assistant. Answer the question based on the given "
            "context. Do not invent information. Answer directly and concisely.\n\n"
            f"Context:\n{document}"
        )
    else:
        prompt = (
            "You are a helpful assistant. Answer the question directly and "
            "concisely. Output only the final answer."
        )
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ]


def generate_answer(
    metanetwork: Any,
    tokenizer: Any,
    device: torch.device,
    messages: list[dict[str, str]],
    lora_dict: Any,
    max_new_tokens: int,
    conversation_max_length: int,
) -> dict[str, str]:
    with torch.no_grad():
        input_enc = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            max_length=conversation_max_length,
            truncation=True,
            return_dict=True,
            padding="max_length",
            enable_thinking=False,
        )
        input_ids = input_enc["input_ids"].to(device)
        attention_mask = input_enc["attention_mask"].to(device)
        outputs = metanetwork.metamodel.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            ignore_mem_token=True,
            loradict=lora_dict,
        )
        new_tokens = outputs[0, input_ids.shape[1] :]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        think, answer = extract_think_and_answer(raw)
        return {"raw_generation": raw, "think": think, "answer": answer}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    import_runtime_deps()
    setup_shine_imports(args.shine_root)
    cfg = load_config(args)
    device = resolve_device(args.device)

    document, doc_metadata = load_design_doc(args.design_doc)
    qa_pairs = load_qa_pairs(args.qa_pairs)

    LOGGER.info("Loaded design doc from %s", args.design_doc)
    LOGGER.info("Loaded %d QA pairs from %s", len(qa_pairs), args.qa_pairs)
    LOGGER.info("Using device: %s", device)

    metanetwork, tokenizer, metalora = load_shine_runtime(cfg, args, device)

    lora_dict = None
    if "shine" in args.conditions:
        LOGGER.info("Generating LoRA dictionary from design document")
        lora_dict = build_lora_dict(
            metanetwork,
            tokenizer,
            document,
            metalora,
            device,
            args.context_max_length,
        )
        if args.save_lora_dict is not None:
            args.save_lora_dict.parent.mkdir(parents=True, exist_ok=True)
            torch.save(lora_dict, args.save_lora_dict)
            LOGGER.info("Saved LoRA dictionary to %s", args.save_lora_dict)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design_doc_path": str(args.design_doc),
        "qa_pairs_path": str(args.qa_pairs),
        "shine_root": str(args.shine_root),
        "config_path": str(args.config),
        "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
        "model_path": str(args.model_path) if args.model_path else str(cfg.model.model_from),
        "conditions": args.conditions,
        "context_max_length": args.context_max_length,
        "conversation_max_length": args.conversation_max_length,
        "max_new_tokens": args.max_new_tokens,
        "doc_metadata": doc_metadata,
    }

    with args.output.open("w", encoding="utf-8") as f:
        for qa in qa_pairs:
            for condition in args.conditions:
                condition_lora = lora_dict if condition == "shine" else None
                messages = messages_for_condition(condition, document, qa["question"])
                generation = generate_answer(
                    metanetwork=metanetwork,
                    tokenizer=tokenizer,
                    device=device,
                    messages=messages,
                    lora_dict=condition_lora,
                    max_new_tokens=args.max_new_tokens,
                    conversation_max_length=args.conversation_max_length,
                )
                record = {
                    "run": run_metadata,
                    "qa_id": qa["qa_id"],
                    "condition": condition,
                    "question": qa["question"],
                    "expected_answer": qa["expected_answer"],
                    "answer": generation["answer"],
                    "think": generation["think"],
                    "raw_generation": generation["raw_generation"],
                    "qa_metadata": qa["metadata"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                LOGGER.info("Wrote %s / %s", qa["qa_id"], condition)

    LOGGER.info("Evaluation complete: %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
