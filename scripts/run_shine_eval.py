#!/usr/bin/env python3
"""Run the initial MemCoder SHINE evaluation from artifact files.

This script adapts the flow from `vendor/SHINE/inference.ipynb` into a CLI:

- load Qwen + the SHINE metanetwork checkpoint(s)
- generate one LoRA dictionary per checkpoint from one design document
- run generated QA pairs under conditions:
  - `naive`:        question only
  - `in_context`:   design document in the prompt
  - `shine`:        expands to every configured checkpoint variant
  - `shine:<label>`: a single named variant
- write one JSONL record per question/condition

Output schema:
  - <output>            : JSONL, one record per (qa, condition); slim metadata only.
  - <output>.meta.json  : sidecar with the full run metadata, joined by `run_id`.
  - the `condition` field is always `naive`, `in_context`, or `shine:<label>`.
    Legacy single-checkpoint runs use the label `default`.

Single-GPU mode: pass `--checkpoint-dir DIR` plus `--device {auto,cuda,cpu}`.
Multi-GPU mode: pass `--qwen-cuda N` plus repeatable
`--shine-checkpoint label=NAME,path=DIR,cuda=M` flags (or
`memcoder_eval.shine_checkpoints` in the YAML).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("memcoder.shine_eval")
torch = None
OmegaConf = None
AutoTokenizer = None
REPO_ROOT = Path(__file__).resolve().parents[1]
CUDA_DEFAULT_DEVICE = 0


def import_runtime_deps() -> None:
    global AutoTokenizer, OmegaConf, torch
    import torch as torch_module
    from omegaconf import OmegaConf as omega_conf
    from transformers import AutoTokenizer as auto_tokenizer

    torch = torch_module
    OmegaConf = omega_conf
    AutoTokenizer = auto_tokenizer


def resolve_repo_path(path: Path | None) -> Path | None:
    """Resolve a path against repo root.

    Rules:
    - `None` stays `None`
    - relative paths are treated as repo-root relative
    - absolute paths stay absolute, even if they do not exist yet
    """
    if path is None:
        return None

    if path.is_absolute():
        return path

    return REPO_ROOT / path


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
        help="Single-checkpoint mode: directory containing metanetwork.pth and metalora.pth.",
    )
    parser.add_argument(
        "--shine-checkpoint",
        action="append",
        default=[],
        metavar="label=NAME,path=DIR[,cuda=N]",
        help=(
            "Multi-checkpoint mode (repeatable). Each occurrence adds a SHINE "
            "checkpoint as its own condition (`shine:<label>`) hosted on its own GPU. "
            "Requires --qwen-cuda."
        ),
    )
    parser.add_argument(
        "--qwen-cuda",
        type=int,
        default=None,
        help="GPU index for the shared Qwen base model when --shine-checkpoint is used.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Override cfg.paths.model_path / cfg.model.model_from / tokenizer_from.",
    )
    parser.add_argument(
        "--save-lora-dict",
        type=Path,
        help=(
            "Optional path to save generated LoRA dictionaries with torch.save. "
            "In multi-checkpoint mode, the variant label is appended to the stem."
        ),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="One or more of: naive | in_context | shine | shine:<label>",
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


def parse_condition_token(token: str) -> tuple[str, str | None]:
    """Parse a condition string into (kind, label).

    Returns (kind, None) for naive | in_context | bare-shine and
    ("shine", label) for shine:<label>.
    """
    if token in ("naive", "in_context"):
        return token, None
    if token == "shine":
        return "shine", None
    if token.startswith("shine:") and len(token) > len("shine:"):
        return "shine", token[len("shine:") :]
    raise ValueError(
        f"Unknown condition: {token!r}; expected naive | in_context | shine | shine:<label>"
    )


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

    value = payload.get("document")
    if isinstance(value, str) and value.strip():
        return value, payload

    raise ValueError(
        f"Could not find design doc text in {path}; expected `document`."
    )


def _answer_from_record(record: dict[str, Any]) -> Any:
    return record.get("answer")


def load_qa_pairs(path: Path) -> list[dict[str, Any]]:
    payload = load_json_or_text(path)
    if isinstance(payload, dict):
        if "qa_pairs" in payload:
            records = payload["qa_pairs"]
        else:
            raise ValueError(f"Could not find `qa_pairs` in {path}")
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


def setup_shine_imports(shine_root: Path | None) -> None:
    if shine_root is None:
        raise ValueError(
            "SHINE root is not set. Pass --shine-root or set memcoder_eval.shine_root in config."
        )
    shine_root = shine_root.resolve()
    if not shine_root.exists():
        raise FileNotFoundError(f"SHINE root does not exist: {shine_root}")
    sys.path.insert(0, str(shine_root))


def _parse_kv_string(spec: str) -> dict[str, str]:
    parts: dict[str, str] = {}
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Expected key=value tokens in {spec!r}")
        key, value = token.split("=", 1)
        parts[key.strip()] = value.strip()
    return parts


def parse_shine_checkpoint_spec(spec: Any) -> dict[str, Any]:
    if isinstance(spec, str):
        parts = _parse_kv_string(spec)
    elif isinstance(spec, dict):
        parts = {str(k): str(v) for k, v in spec.items()}
    else:
        try:
            parts = {str(k): str(v) for k, v in dict(spec).items()}
        except Exception as e:
            raise ValueError(f"Cannot parse shine checkpoint spec: {spec!r}") from e

    if "label" not in parts:
        raise ValueError(f"shine checkpoint spec missing 'label': {spec!r}")
    if "path" not in parts:
        raise ValueError(f"shine checkpoint spec missing 'path': {spec!r}")

    out: dict[str, Any] = {
        "label": parts["label"],
        "path": resolve_repo_path(Path(parts["path"])),
    }
    if "cuda" in parts and parts["cuda"] != "":
        out["cuda"] = int(parts["cuda"])
    return out


def load_config(args: argparse.Namespace, require_eval_paths: bool = True):
    args.config = resolve_repo_path(args.config)
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    eval_cfg = cfg.get("memcoder_eval", {})

    def cfg_path(name: str) -> Path | None:
        value = eval_cfg.get(name)
        return resolve_repo_path(Path(str(value))) if value is not None else None

    args.design_doc = args.design_doc or cfg_path("design_doc")
    args.qa_pairs = args.qa_pairs or cfg_path("qa_pairs")
    args.output = args.output or cfg_path("output")
    args.shine_root = args.shine_root or cfg_path("shine_root") or Path("vendor/SHINE")
    args.checkpoint_dir = args.checkpoint_dir or cfg_path("checkpoint_dir")
    args.model_path = args.model_path or cfg_path("model_path")
    args.save_lora_dict = args.save_lora_dict or cfg_path("save_lora_dict")
    args.design_doc = resolve_repo_path(args.design_doc)
    args.qa_pairs = resolve_repo_path(args.qa_pairs)
    args.output = resolve_repo_path(args.output)
    args.shine_root = resolve_repo_path(args.shine_root)
    args.checkpoint_dir = resolve_repo_path(args.checkpoint_dir)
    args.model_path = resolve_repo_path(args.model_path)
    args.save_lora_dict = resolve_repo_path(args.save_lora_dict)
    args.conditions = args.conditions or list(
        eval_cfg.get("conditions", ["naive", "in_context", "shine"])
    )
    args.device = args.device or str(eval_cfg.get("device", cfg.run.get("device", "auto")))
    args.context_max_length = args.context_max_length or int(
        eval_cfg.get("context_max_length", cfg.test.context_max_length)
    )
    if args.context_max_length is None:
        args.context_max_length = int(eval_cfg.get("context_max_length", cfg.test.context_max_length))
    if args.conversation_max_length is None:
        args.conversation_max_length = int(eval_cfg.get("conversation_max_length", cfg.test.conversation_max_length))
    if args.max_new_tokens is None:
        args.max_new_tokens = int(eval_cfg.get("max_new_tokens", cfg.test.max_new_tokens))
    if args.seed is None:
        args.seed = int(eval_cfg.get("seed", cfg.run.get("seed", 42)))
    if args.qwen_cuda is None and "qwen_cuda" in eval_cfg:
        args.qwen_cuda = int(eval_cfg.get("qwen_cuda"))

    yaml_shine = eval_cfg.get("shine_checkpoints", None)
    if not args.shine_checkpoint and yaml_shine:
        args.shine_checkpoint = list(yaml_shine)

    required_paths = ("design_doc", "qa_pairs", "output") if require_eval_paths else ("design_doc",)
    missing = [name for name in required_paths if getattr(args, name) is None]
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
    else:
        cfg.paths.model_path = str(resolve_repo_path(Path(str(cfg.paths.model_path))))
        cfg.model.model_from = str(resolve_repo_path(Path(str(cfg.model.model_from))))
        cfg.model.tokenizer_from = str(resolve_repo_path(Path(str(cfg.model.tokenizer_from))))

    cfg.run.seed = args.seed
    cfg.test.context_max_length = args.context_max_length
    cfg.test.conversation_max_length = args.conversation_max_length
    cfg.test.max_new_tokens = args.max_new_tokens
    return cfg


def resolve_shine_specs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], bool]:
    """Return (specs, multi_mode).

    Multi-mode: each spec carries its own hyper_device.
    Legacy single-mode: one spec with label='default' and hyper_device=None
    (filled in later from qwen_device).
    """
    multi = [parse_shine_checkpoint_spec(s) for s in args.shine_checkpoint]
    if multi:
        if args.checkpoint_dir is not None:
            LOGGER.warning(
                "Both --checkpoint-dir and --shine-checkpoint are configured; "
                "--checkpoint-dir is being ignored."
            )
            args.checkpoint_dir = None
        if args.qwen_cuda is None:
            raise ValueError(
                "--qwen-cuda is required when --shine-checkpoint is used; "
                "the shared Qwen base must live on a GPU distinct from each hypernetwork."
            )
        used_cuda = {args.qwen_cuda}
        seen_labels: set[str] = set()
        for spec in multi:
            if "cuda" not in spec:
                raise ValueError(
                    f"shine checkpoint {spec['label']!r} is missing 'cuda='. Each "
                    "hypernetwork must live on its own GPU in multi-checkpoint mode."
                )
            if spec["cuda"] in used_cuda:
                raise ValueError(
                    f"shine checkpoint {spec['label']!r} requests cuda:{spec['cuda']} "
                    "which is already in use by another component."
                )
            if spec["label"] in seen_labels:
                raise ValueError(f"duplicate shine checkpoint label: {spec['label']!r}")
            used_cuda.add(spec["cuda"])
            seen_labels.add(spec["label"])
            spec["hyper_device"] = torch.device(f"cuda:{spec['cuda']}")
        return multi, True

    if args.checkpoint_dir is not None:
        return (
            [{"label": "default", "path": args.checkpoint_dir, "hyper_device": None}],
            False,
        )

    return [], False


def resolve_qwen_device(args: argparse.Namespace, multi_mode: bool) -> "torch.device":
    if multi_mode:
        return torch.device(f"cuda:{args.qwen_cuda}")
    requested = args.device
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is unavailable")
        return torch.device(f"cuda:{CUDA_DEFAULT_DEVICE}")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{CUDA_DEFAULT_DEVICE}" if torch.cuda.is_available() else "cpu")


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


def _bridge_generate_lora_dict(metanetwork: Any, qwen_device: "torch.device", hyper_device: "torch.device") -> None:
    """Patch metanetwork.generate_lora_dict so memory_states/plain_output bridge devices.

    No-op when both devices are equal (single-GPU legacy path).
    """
    if qwen_device == hyper_device:
        return

    def generate_lora_dict_split(self, evidence_ids, evidence_attention_mask, metalora,
                                 use_gradient_checkpoint=False, return_plain=False):
        evidence_ids = evidence_ids.to(qwen_device)
        evidence_attention_mask = evidence_attention_mask.to(qwen_device)
        outputs = self.metamodel(
            input_ids=evidence_ids,
            attention_mask=evidence_attention_mask,
            loradict=metalora,
            use_gradient_checkpoint=use_gradient_checkpoint,
        )
        memory_states = outputs.memory_states.to(hyper_device)
        plain_output = self.metanetwork(memory_states)
        plain_output = plain_output.to(qwen_device)
        loradict = self.metamodel.generate_lora_dict(
            self.lora_r, scale=self.scale, plain_tensor=plain_output
        )
        return (loradict, plain_output) if return_plain else loradict

    metanetwork.generate_lora_dict = types.MethodType(generate_lora_dict_split, metanetwork)


def load_base_runtime(cfg: Any, qwen_device: "torch.device") -> tuple[Any, Any]:
    """Load the shared Qwen base + tokenizer onto qwen_device, in eval mode."""
    from utils.myinit import _import_class
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

    LOGGER.info("Loading Qwen model from %s onto %s", cfg.model.model_from, qwen_device)
    metamodel = meta_model_cls.from_pretrained(cfg.model.model_from, config=config)
    metamodel.reset_mem_tokens()
    metamodel.resize_token_embeddings(len(tokenizer))
    metamodel.to(qwen_device)
    metamodel.eval()
    for param in metamodel.parameters():
        param.requires_grad = False

    return metamodel, tokenizer


def attach_shine_variant(
    metamodel: Any,
    cfg: Any,
    label: str,
    checkpoint_dir: Path | None,
    qwen_device: "torch.device",
    hyper_device: "torch.device",
    allow_random: bool,
) -> dict[str, Any]:
    """Build a Metanetwork wrapping the shared metamodel; place its hypernet on hyper_device.

    Caches the variant's mem_tokens on CPU so the caller can rebind them to the
    shared metamodel right before generate_lora_dict — making correctness
    independent of variant load order.
    """
    from metanetwork_family import Metanetwork
    from utils.mysaveload import move_to_device_and_change_into_leaf

    LOGGER.info(
        "Initializing SHINE metanetwork variant %r (hypernet on %s)",
        label,
        hyper_device,
    )
    metanetwork = Metanetwork(metamodel, cfg, metamodel.lora_params_numel(cfg.model.lora_r))
    metanetwork.metanetwork.to(hyper_device)
    _bridge_generate_lora_dict(metanetwork, qwen_device, hyper_device)
    metanetwork.eval()

    metalora = None
    cached_mem_tokens = None
    if checkpoint_dir is not None:
        ckpt = Path(checkpoint_dir)
        LOGGER.info("Loading SHINE checkpoint %r from %s", label, ckpt)
        if metanetwork.metamodel.model.use_mem_token:
            cached_mem_tokens = torch.load(
                os.path.join(ckpt, "mem_tokens.pt"),
                map_location="cpu",
                weights_only=False,
            )
            target_shape = metanetwork.metamodel.model.mem_tokens.shape
            if cached_mem_tokens.shape != target_shape:
                raise ValueError(
                    f"mem_tokens shape mismatch in {ckpt}: saved "
                    f"{cached_mem_tokens.shape} vs model {target_shape}"
                )
        metanetwork.metanetwork.load_state_dict(
            torch.load(
                os.path.join(ckpt, "metanetwork.pth"),
                weights_only=False,
                map_location="cpu",
            )
        )
        metalora_cpu = torch.load(
            os.path.join(ckpt, "metalora.pth"),
            map_location="cpu",
            weights_only=False,
        )
        metalora = move_to_device_and_change_into_leaf(metalora_cpu, qwen_device)
    elif not allow_random:
        raise ValueError(
            f"`shine` variant {label!r} requires a checkpoint directory unless "
            "--allow-random-metanetwork is set."
        )
    else:
        LOGGER.warning("Variant %r has no checkpoint; using initialized weights", label)

    return {
        "label": label,
        "metanetwork": metanetwork,
        "metalora": metalora,
        "mem_tokens_cpu": cached_mem_tokens,
        "hyper_device": hyper_device,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
    }


def bind_variant_mem_tokens(metamodel: Any, variant: dict[str, Any], qwen_device: "torch.device") -> None:
    """Install this variant's mem_tokens on the shared metamodel."""
    if variant.get("mem_tokens_cpu") is None:
        return
    metamodel.model.mem_tokens = torch.nn.Parameter(
        variant["mem_tokens_cpu"].to(qwen_device), requires_grad=False
    )


def release_variant_runtime(variant: dict[str, Any]) -> None:
    """Free the hypernet/metalora once we have its lora_dict; keep label+device for metadata."""
    variant["metanetwork"] = None
    variant["metalora"] = None
    variant["mem_tokens_cpu"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_lora_dict(
    metanetwork: Any,
    tokenizer: Any,
    document: str,
    metalora: Any,
    qwen_device: "torch.device",
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
        evidence_ids = enc["input_ids"].to(qwen_device)
        evidence_attention_mask = enc["attention_mask"].to(qwen_device)
        return metanetwork.generate_lora_dict(evidence_ids, evidence_attention_mask, metalora)


def build_messages(question: str, document: str | None = None) -> list[dict[str, str]]:
    if document is not None:
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
    metamodel: Any,
    tokenizer: Any,
    qwen_device: "torch.device",
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
        input_ids = input_enc["input_ids"].to(qwen_device)
        attention_mask = input_enc["attention_mask"].to(qwen_device)
        outputs = metamodel.generate(
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


def _save_lora_dict(base_path: Path, label: str, lora_dict: Any, multi: bool) -> Path:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    out = base_path.with_name(f"{base_path.stem}__{label}{base_path.suffix}") if multi else base_path
    torch.save(lora_dict, out)
    LOGGER.info("Saved LoRA dictionary (%s) to %s", label, out)
    return out


def _artifact_relative(path: Path, anchor: Path) -> str:
    try:
        return path.resolve().relative_to(anchor.resolve()).as_posix()
    except ValueError:
        return str(path)


def _difficulty_dir_from_doc_path(design_doc_path: Path) -> Path | None:
    if design_doc_path.parent.name != "docs":
        return None
    return design_doc_path.parent.parent


def _repo_dir_from_doc_path(design_doc_path: Path) -> Path | None:
    difficulty_dir = _difficulty_dir_from_doc_path(design_doc_path)
    if difficulty_dir is None:
        return None
    return difficulty_dir.parent


def update_ledger(
    *,
    design_doc_path: Path,
    qa_pairs_path: Path,
    doc_metadata: dict[str, Any],
    saved_lora_paths: list[tuple[str, Path]],
) -> Path | None:
    """Refresh the repo-level ledger.json with canonical artifact paths."""
    if not saved_lora_paths:
        return None

    difficulty_dir = _difficulty_dir_from_doc_path(design_doc_path)
    repo_dir = _repo_dir_from_doc_path(design_doc_path)
    if difficulty_dir is None or repo_dir is None:
        LOGGER.warning("Could not infer artifact repo directory from %s", design_doc_path)
        return None

    document_id = doc_metadata.get("document_id") or design_doc_path.stem
    difficulty = doc_metadata.get("difficulty") or difficulty_dir.name
    ledger_path = repo_dir / "ledger.json"

    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if not isinstance(ledger, dict):
            raise ValueError(f"Ledger must be a JSON object: {ledger_path}")
    else:
        ledger = {}

    documents = ledger.get("documents")
    if not isinstance(documents, dict):
        documents = {}

    files: dict[str, Any] = {
        "doc": _artifact_relative(design_doc_path, repo_dir),
        "doc_embedding": None,
        "qa": _artifact_relative(qa_pairs_path, repo_dir),
        "qa_examples": None,
    }
    if len(saved_lora_paths) == 1:
        files["lora"] = _artifact_relative(saved_lora_paths[0][1], repo_dir)
    else:
        files["lora"] = None
        files["lora_variants"] = {
            label: _artifact_relative(path, repo_dir)
            for label, path in saved_lora_paths
        }

    entry: dict[str, Any] = {
        "document_id": document_id,
        "difficulty": difficulty,
        "topic": doc_metadata.get("topic"),
        "files": files,
    }
    if doc_metadata.get("topic_slug"):
        entry["topic_slug"] = doc_metadata["topic_slug"]

    documents[document_id] = entry
    ledger = {"documents": documents}
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Updated artifact ledger at %s", ledger_path)
    return ledger_path


def select_variants_to_load(
    parsed_conditions: list[tuple[str, str | None]],
    shine_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pick the subset of specs we actually need given the parsed conditions."""
    shine_kinds = [(kind, label) for kind, label in parsed_conditions if kind == "shine"]
    if not shine_kinds:
        return []

    available = {s["label"] for s in shine_specs}
    bare_shine = any(label is None for _, label in shine_kinds)
    explicit_labels = {label for _, label in shine_kinds if label is not None}

    unknown = explicit_labels - available
    if unknown:
        raise ValueError(
            f"Conditions reference unknown shine labels: {sorted(unknown)}; "
            f"configured: {sorted(available)}"
        )

    if bare_shine:
        return list(shine_specs)
    return [s for s in shine_specs if s["label"] in explicit_labels]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    import_runtime_deps()
    cfg = load_config(args)
    setup_shine_imports(args.shine_root)

    parsed_conditions = [parse_condition_token(c) for c in args.conditions]

    shine_specs, multi_mode = resolve_shine_specs(args)
    qwen_device = resolve_qwen_device(args, multi_mode)
    for spec in shine_specs:
        if spec["hyper_device"] is None:
            spec["hyper_device"] = qwen_device

    if torch.cuda.is_available() and qwen_device.type == "cuda":
        # set_device requires an explicit CUDA index.
        torch.cuda.set_device(
            qwen_device.index if qwen_device.index is not None else CUDA_DEFAULT_DEVICE
        )

    document, doc_metadata = load_design_doc(args.design_doc)
    qa_pairs = load_qa_pairs(args.qa_pairs)

    LOGGER.info("Loaded design doc from %s", args.design_doc)
    LOGGER.info("Loaded %d QA pairs from %s", len(qa_pairs), args.qa_pairs)
    if multi_mode:
        LOGGER.info(
            "Multi-checkpoint mode: Qwen on %s, hypernets on %s",
            qwen_device,
            ", ".join(f"{s['label']}@{s['hyper_device']}" for s in shine_specs),
        )
    else:
        LOGGER.info("Single-checkpoint / single-GPU mode on %s", qwen_device)

    metamodel, tokenizer = load_base_runtime(cfg, qwen_device)

    # Build LoRA dicts only for variants we actually need.
    # Invariant: each variant's LoRA dict must be built while THAT variant's
    # mem_tokens are bound to the shared metamodel. We rebind right before each
    # build so this is order-independent. After the build, the LoRA weights are
    # baked into a tensor dict and QA generation uses ignore_mem_token=True, so
    # the metamodel's mem_tokens state no longer matters.
    variants_to_load = select_variants_to_load(parsed_conditions, shine_specs)
    shine_runs: list[dict[str, Any]] = []
    saved_lora_paths: list[tuple[str, Path]] = []
    for spec in variants_to_load:
        variant = attach_shine_variant(
            metamodel=metamodel,
            cfg=cfg,
            label=spec["label"],
            checkpoint_dir=spec["path"],
            qwen_device=qwen_device,
            hyper_device=spec["hyper_device"],
            allow_random=args.allow_random_metanetwork,
        )
        bind_variant_mem_tokens(metamodel, variant, qwen_device)
        LOGGER.info("Generating LoRA dictionary for variant %r", spec["label"])
        variant["lora_dict"] = build_lora_dict(
            variant["metanetwork"],
            tokenizer,
            document,
            variant["metalora"],
            qwen_device,
            args.context_max_length,
        )
        if args.save_lora_dict is not None:
            saved_path = _save_lora_dict(
                args.save_lora_dict,
                spec["label"],
                variant["lora_dict"],
                multi=multi_mode,
            )
            saved_lora_paths.append((spec["label"], saved_path))
        # Free hypernet + metalora; keep only what QA generation and metadata need.
        release_variant_runtime(variant)
        shine_runs.append(variant)

    update_ledger(
        design_doc_path=args.design_doc,
        qa_pairs_path=args.qa_pairs,
        doc_metadata=doc_metadata,
        saved_lora_paths=saved_lora_paths,
    )

    # Expand conditions for the QA loop. Always emits "shine:<label>" so the
    # output schema is uniform across single- and multi-mode runs.
    expanded: list[tuple[str, dict[str, Any] | None]] = []
    for kind, label in parsed_conditions:
        if kind == "shine":
            if label is None:
                for run in shine_runs:
                    expanded.append((f"shine:{run['label']}", run))
            else:
                run = next(r for r in shine_runs if r["label"] == label)
                expanded.append((f"shine:{label}", run))
        else:
            expanded.append((kind, None))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design_doc_path": str(args.design_doc),
        "qa_pairs_path": str(args.qa_pairs),
        "shine_root": str(args.shine_root),
        "config_path": str(args.config),
        "qwen_device": str(qwen_device),
        "model_path": str(args.model_path) if args.model_path else str(cfg.model.model_from),
        "conditions": args.conditions,
        "shine_checkpoints": [
            {
                "label": v["label"],
                "checkpoint_dir": v["checkpoint_dir"],
                "hyper_device": str(v["hyper_device"]),
            }
            for v in shine_runs
        ],
        "context_max_length": args.context_max_length,
        "conversation_max_length": args.conversation_max_length,
        "max_new_tokens": args.max_new_tokens,
        "doc_metadata": doc_metadata,
    }

    sidecar_path = args.output.with_name(f"{args.output.name}.meta.json")
    sidecar_path.write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Wrote run metadata sidecar to %s", sidecar_path)

    with args.output.open("w", encoding="utf-8") as f:
        for qa in qa_pairs:
            for condition_name, run in expanded:
                lora_dict = run["lora_dict"] if run is not None else None
                doc_for_prompt = document if condition_name == "in_context" else None
                messages = build_messages(qa["question"], doc_for_prompt)
                generation = generate_answer(
                    metamodel=metamodel,
                    tokenizer=tokenizer,
                    qwen_device=qwen_device,
                    messages=messages,
                    lora_dict=lora_dict,
                    max_new_tokens=args.max_new_tokens,
                    conversation_max_length=args.conversation_max_length,
                )
                record = {
                    "run_id": run_id,
                    "qa_id": qa["qa_id"],
                    "condition": condition_name,
                    "shine_label": run["label"] if run is not None else None,
                    "shine_checkpoint_dir": run["checkpoint_dir"] if run is not None else None,
                    "question": qa["question"],
                    "expected_answer": qa["expected_answer"],
                    "answer": generation["answer"],
                    "think": generation["think"],
                    "raw_generation": generation["raw_generation"],
                    "qa_metadata": qa["metadata"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                LOGGER.info("Wrote %s / %s", qa["qa_id"], condition_name)

    LOGGER.info("Evaluation complete: %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
