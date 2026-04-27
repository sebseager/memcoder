#!/usr/bin/env python3
"""Generate Kilo topic LoRAs, compose them, and compare composed vs individual answers."""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_shine_eval as shine_eval


LOGGER = logging.getLogger("memcoder.lora_composition_eval")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = Path("artifacts/antirez__kilo/easy")
DEFAULT_DOCS = [
    DEFAULT_ARTIFACT_DIR / "docs" / "overview_purpose_1.json",
    DEFAULT_ARTIFACT_DIR / "docs" / "raw_terminal_input_1.json",
    DEFAULT_ARTIFACT_DIR / "docs" / "rows_editing_persistence_1.json",
]
DEFAULT_GENERATE_DOCS = DEFAULT_DOCS[1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=Path("config/shine_eval_demo.yaml"), type=Path)
    parser.add_argument(
        "--doc",
        action="append",
        type=Path,
        default=None,
        help="Document to include in the composition experiment. Repeatable.",
    )
    parser.add_argument(
        "--generate-doc",
        action="append",
        type=Path,
        default=None,
        help="Document to pass through generate_shine_lora.py before evaluation. Repeatable.",
    )
    parser.add_argument(
        "--qa-pairs",
        action="append",
        type=Path,
        default=None,
        help=(
            "QA file to evaluate instead of each document's default sibling QA file. "
            "Repeatable; records may include required_document_ids for cross-document questions."
        ),
    )
    parser.add_argument(
        "--lora-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "loras",
        help="Directory containing/generated .pt LoRA dictionaries.",
    )
    parser.add_argument(
        "--lora-label",
        help="Use <document_id>__<label>.pt instead of <document_id>.pt, for multi-checkpoint artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "lora_composition_results.jsonl",
        help="JSONL answer records for individual and composed LoRA runs.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Summary JSON path. Defaults to <output>.summary.json.",
    )
    parser.add_argument(
        "--composition-method",
        choices=["rank_average", "rank_sum"],
        default="rank_average",
        help=(
            "How to compose LoRA dictionaries. rank_average concatenates ranks but "
            "scales each adapter delta by 1/N; rank_sum preserves the original "
            "unscaled concat behavior."
        ),
    )
    parser.add_argument(
        "--composition-scale",
        type=float,
        default=1.0,
        help="Additional multiplier for the composed adapter delta.",
    )
    parser.add_argument("--skip-generate-loras", action="store_true")
    parser.add_argument(
        "--force-loras",
        action="store_true",
        help="Regenerate requested LoRA artifacts even if their files already exist.",
    )
    parser.add_argument("--shine-root", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--qwen-cuda", type=int, default=None)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--context-max-length", type=int)
    parser.add_argument("--conversation-max-length", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--allow-random-metanetwork", action="store_true")
    parser.add_argument(
        "--shine-checkpoint",
        action="append",
        default=[],
        metavar="label=NAME,path=DIR[,cuda=N]",
        help="Forwarded to generate_shine_lora.py and config loading.",
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


def repo_path(path: Path) -> Path:
    return shine_eval.resolve_repo_path(path)


def document_id_from_doc(path: Path) -> str:
    payload = shine_eval.load_json_or_text(path)
    if isinstance(payload, dict) and isinstance(payload.get("document_id"), str):
        return payload["document_id"]
    return path.stem


def qa_path_for_doc(doc_path: Path, document_id: str) -> Path:
    if doc_path.parent.name == "docs":
        return doc_path.parent.parent / "qas" / f"{document_id}.json"
    raise ValueError(f"Cannot infer QA path for non-artifact document: {doc_path}")


def lora_path_for_doc(lora_dir: Path, document_id: str, label: str | None) -> Path:
    stem = f"{document_id}__{label}" if label else document_id
    return lora_dir / f"{stem}.pt"


def _string_list(value: Any, field_name: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        items: list[str] = []
        for idx, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{field_name}[{idx}] must be a non-empty string")
            items.append(item.strip())
        return items
    raise ValueError(f"{field_name} must be a string or list of strings")


def required_document_ids_for_qa(qa: dict[str, Any], fallback_document_id: str) -> list[str]:
    metadata = qa["metadata"]
    required_ids = (
        _string_list(metadata.get("required_document_ids"), "required_document_ids")
        or _string_list(metadata.get("document_ids"), "document_ids")
        or _string_list(metadata.get("documents"), "documents")
    )
    if required_ids:
        return required_ids

    document_id = metadata.get("document_id")
    if isinstance(document_id, str) and document_id.strip():
        return [document_id.strip()]
    return [fallback_document_id]


def qa_scope_id(qa: dict[str, Any], required_ids: list[str]) -> str:
    metadata = qa["metadata"]
    for key in ("composition_scope", "scope_id", "document_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "+".join(required_ids)


def load_default_qa_items(doc_ids: dict[Path, str]) -> list[dict[str, Any]]:
    qa_items: list[dict[str, Any]] = []
    for doc_path, document_id in doc_ids.items():
        _document, doc_metadata = shine_eval.load_design_doc(doc_path)
        qa_path = qa_path_for_doc(doc_path, document_id)
        for qa in shine_eval.load_qa_pairs(qa_path):
            qa_items.append(
                {
                    "document_id": document_id,
                    "topic": doc_metadata.get("topic"),
                    "qa": qa,
                    "qa_pairs_path": str(qa_path),
                    "individual_document_ids": [document_id],
                }
            )
    return qa_items


def load_override_qa_items(qa_paths: list[Path], doc_ids: dict[Path, str]) -> list[dict[str, Any]]:
    available_doc_ids = set(doc_ids.values())
    fallback_document_id = next(iter(doc_ids.values()))
    qa_items: list[dict[str, Any]] = []

    for qa_path in qa_paths:
        payload = shine_eval.load_json_or_text(qa_path)
        default_topic = payload.get("topic") if isinstance(payload, dict) else None
        for qa in shine_eval.load_qa_pairs(qa_path):
            required_ids = required_document_ids_for_qa(qa, fallback_document_id)
            unknown_ids = sorted(set(required_ids) - available_doc_ids)
            if unknown_ids:
                raise ValueError(
                    f"{qa_path} / {qa['qa_id']} requires document IDs not included by --doc: "
                    f"{', '.join(unknown_ids)}"
                )
            qa_items.append(
                {
                    "document_id": qa_scope_id(qa, required_ids),
                    "topic": qa["metadata"].get("topic") or default_topic,
                    "qa": qa,
                    "qa_pairs_path": str(qa_path),
                    "individual_document_ids": required_ids,
                }
            )
    return qa_items


def composition_weights(count: int, method: str, scale: float) -> list[float]:
    if count <= 0:
        raise ValueError("At least one LoRA is required for composition")
    if method == "rank_sum":
        return [scale] * count
    if method == "rank_average":
        return [scale / count] * count
    raise ValueError(f"Unknown composition method: {method}")


def compose_lora_dicts(lora_dicts: list[Any], weights: list[float]) -> Any:
    """Compose LoRAs by rank-concat while weighting each adapter's output delta.

    LoraLinear computes `(x @ A) @ B + C`. Concatenating A/B ranks sums deltas,
    so averaging means scaling one side of each adapter pair, plus C, by 1/N.
    """
    if len(lora_dicts) != len(weights):
        raise ValueError("lora_dicts and weights must have the same length")
    if not lora_dicts:
        raise ValueError("Cannot compose an empty LoRA list")

    def _compose(nodes: list[Any], path: str) -> Any:
        first = nodes[0]
        if isinstance(first, dict) and "A" in first and "B" in first:
            if not all(isinstance(node, dict) and "A" in node and "B" in node for node in nodes):
                raise TypeError(f"{path}: all leaves must be LoRA A/B dictionaries")

            a_tensors = [node["A"] for node in nodes]
            b_tensors = [node["B"] for node in nodes]
            c_tensors = [node.get("C", None) for node in nodes]

            for idx, (a_tensor, b_tensor) in enumerate(zip(a_tensors, b_tensors, strict=True)):
                if a_tensor is None or b_tensor is None:
                    raise ValueError(f"{path}: A/B cannot be None for adapter {idx}")
                if a_tensor.shape[0] != a_tensors[0].shape[0]:
                    raise ValueError(f"{path}.A: Lb mismatch for adapter {idx}")
                if b_tensor.shape[0] != b_tensors[0].shape[0]:
                    raise ValueError(f"{path}.B: Lb mismatch for adapter {idx}")
                if a_tensor.shape[1] != a_tensors[0].shape[1]:
                    raise ValueError(f"{path}.A: in_features mismatch for adapter {idx}")
                if b_tensor.shape[2] != b_tensors[0].shape[2]:
                    raise ValueError(f"{path}.B: out_features mismatch for adapter {idx}")
                if a_tensor.shape[2] != b_tensor.shape[1]:
                    raise ValueError(f"{path}: rank mismatch inside adapter {idx}")

            has_c = [c_tensor is not None for c_tensor in c_tensors]
            if any(has_c) and not all(has_c):
                raise ValueError(f"{path}.C: either all adapters must have C or none may have C")

            return {
                "A": shine_eval.torch.cat(
                    [a_tensor * weight for a_tensor, weight in zip(a_tensors, weights, strict=True)],
                    dim=2,
                ),
                "B": shine_eval.torch.cat(b_tensors, dim=1),
                "C": None
                if not any(has_c)
                else sum(c_tensor * weight for c_tensor, weight in zip(c_tensors, weights, strict=True)),
            }

        if isinstance(first, dict):
            keys = set(first.keys())
            for idx, node in enumerate(nodes):
                if not isinstance(node, dict):
                    raise TypeError(f"{path}: adapter {idx} is not a dictionary")
                if set(node.keys()) != keys:
                    raise ValueError(f"{path}: key mismatch for adapter {idx}")
            return {
                key: _compose(
                    [node[key] for node in nodes],
                    path=f"{path}.{key}" if path else str(key),
                )
                for key in first.keys()
            }

        raise TypeError(f"{path}: unsupported LoRA node type {type(first)}")

    return _compose(lora_dicts, "")


def forward_arg(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def generate_lora(doc_path: Path, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_shine_lora.py"),
        "--config",
        str(args.config),
        "--design-doc",
        str(doc_path),
        "--lora-dir",
        str(args.lora_dir),
    ]
    forward_arg(cmd, "--shine-root", args.shine_root)
    forward_arg(cmd, "--checkpoint-dir", args.checkpoint_dir)
    forward_arg(cmd, "--qwen-cuda", args.qwen_cuda)
    forward_arg(cmd, "--model-path", args.model_path)
    forward_arg(cmd, "--device", args.device)
    forward_arg(cmd, "--context-max-length", args.context_max_length)
    forward_arg(cmd, "--seed", args.seed)
    for spec in args.shine_checkpoint:
        cmd.extend(["--shine-checkpoint", spec])
    for override in args.overrides:
        cmd.extend(["--set", override])
    if args.allow_random_metanetwork:
        cmd.append("--allow-random-metanetwork")

    LOGGER.info("Generating LoRA via: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def normalize_answer(text: Any) -> str:
    text = "" if text is None else str(text).lower()
    text = re.sub(r"`", "", text)
    text = re.sub(r"[^a-z0-9_]+", " ", text)
    return " ".join(text.split())


def token_f1(expected: Any, answer: Any) -> float:
    expected_tokens = normalize_answer(expected).split()
    answer_tokens = normalize_answer(answer).split()
    if not expected_tokens and not answer_tokens:
        return 1.0
    if not expected_tokens or not answer_tokens:
        return 0.0
    remaining = answer_tokens.copy()
    overlap = 0
    for token in expected_tokens:
        if token in remaining:
            remaining.remove(token)
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(answer_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(expected: Any, answer: Any) -> dict[str, Any]:
    expected_norm = normalize_answer(expected)
    answer_norm = normalize_answer(answer)
    contains = bool(expected_norm and (expected_norm in answer_norm or answer_norm in expected_norm))
    return {
        "answer_norm": answer_norm,
        "expected_norm": expected_norm,
        "exact_or_contains": contains,
        "token_f1": token_f1(expected, answer),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_doc_condition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_qa: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for record in records:
        by_condition[record["condition"]].append(record)
        by_doc_condition[(record["document_id"], record["condition"])].append(record)
        by_qa[record["qa_id"]][record["condition"]] = record

    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"count": 0, "avg_token_f1": 0.0, "exact_or_contains_rate": 0.0}
        return {
            "count": len(items),
            "avg_token_f1": sum(item["scores"]["token_f1"] for item in items) / len(items),
            "exact_or_contains_rate": sum(1 for item in items if item["scores"]["exact_or_contains"]) / len(items),
        }

    comparisons = []
    for qa_id, runs in sorted(by_qa.items()):
        composition = runs.get("composition")
        individual_runs = [
            run
            for condition, run in runs.items()
            if condition == "individual" or condition.startswith("individual:")
        ]
        if not individual_runs or composition is None:
            continue
        best_individual = max(individual_runs, key=lambda run: run["scores"]["token_f1"])
        delta = composition["scores"]["token_f1"] - best_individual["scores"]["token_f1"]
        comparisons.append(
            {
                "qa_id": qa_id,
                "document_id": composition["document_id"],
                "best_individual_condition": best_individual["condition"],
                "best_individual_adapter_document_id": best_individual.get("adapter_document_id"),
                "individual_token_f1": best_individual["scores"]["token_f1"],
                "composition_token_f1": composition["scores"]["token_f1"],
                "delta_token_f1": delta,
                "composition_changed_answer": normalize_answer(best_individual["answer"])
                != normalize_answer(composition["answer"]),
            }
        )

    individual_items = [
        item
        for condition, items in by_condition.items()
        if condition == "individual" or condition.startswith("individual:")
        for item in items
    ]
    individual_avg = aggregate(individual_items)["avg_token_f1"]
    composition_avg = aggregate(by_condition.get("composition", []))["avg_token_f1"]
    return {
        "conditions": {condition: aggregate(items) for condition, items in sorted(by_condition.items())},
        "documents": {
            f"{doc_id}:{condition}": aggregate(items)
            for (doc_id, condition), items in sorted(by_doc_condition.items())
        },
        "comparisons": comparisons,
        "composition_minus_individual_avg_token_f1": composition_avg - individual_avg,
        "composition_effective_by_token_f1": composition_avg >= individual_avg * 0.95,
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    docs = [repo_path(path) for path in (args.doc or DEFAULT_DOCS)]
    generate_docs = [repo_path(path) for path in (args.generate_doc or DEFAULT_GENERATE_DOCS)]
    qa_paths = [repo_path(path) for path in args.qa_pairs] if args.qa_pairs else None
    args.lora_dir = repo_path(args.lora_dir)
    args.output = repo_path(args.output)
    args.summary_output = repo_path(args.summary_output) if args.summary_output else args.output.with_name(f"{args.output.name}.summary.json")

    doc_ids = {doc_path: document_id_from_doc(doc_path) for doc_path in docs}

    if not args.skip_generate_loras:
        generated: set[Path] = set()
        for doc_path in generate_docs:
            if doc_path not in generated:
                generate_lora(doc_path, args)
                generated.add(doc_path)
        for doc_path, document_id in doc_ids.items():
            expected_lora = lora_path_for_doc(args.lora_dir, document_id, args.lora_label)
            if (args.force_loras or not expected_lora.exists()) and doc_path not in generated:
                generate_lora(doc_path, args)
                generated.add(doc_path)

    args.design_doc = docs[0]
    args.qa_pairs = qa_path_for_doc(docs[0], doc_ids[docs[0]])
    args.conditions = ["shine"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.save_lora_dict = None

    shine_eval.import_runtime_deps()
    cfg = shine_eval.load_config(args)
    shine_eval.setup_shine_imports(args.shine_root)

    if args.qwen_cuda is not None:
        qwen_device = shine_eval.torch.device(f"cuda:{args.qwen_cuda}")
    else:
        qwen_device = shine_eval.resolve_qwen_device(args, multi_mode=False)
    if shine_eval.torch.cuda.is_available() and qwen_device.type == "cuda":
        shine_eval.torch.cuda.set_device(
            qwen_device.index if qwen_device.index is not None else shine_eval.CUDA_DEFAULT_DEVICE
        )

    metamodel, tokenizer = shine_eval.load_base_runtime(cfg, qwen_device)

    loras: dict[str, Any] = {}
    for doc_path, document_id in doc_ids.items():
        path = lora_path_for_doc(args.lora_dir, document_id, args.lora_label)
        if not path.exists():
            raise FileNotFoundError(f"Missing LoRA artifact for {document_id}: {path}")
        lora = shine_eval.torch.load(path, map_location=qwen_device, weights_only=False)
        loras[document_id] = lora
    weights = composition_weights(
        count=len(loras),
        method=args.composition_method,
        scale=args.composition_scale,
    )
    composed_lora = compose_lora_dicts(list(loras.values()), weights)
    LOGGER.info(
        "Composed %d LoRAs with %s weights=%s",
        len(loras),
        args.composition_method,
        ", ".join(f"{weight:.4g}" for weight in weights),
    )

    qa_items = load_override_qa_items(qa_paths, doc_ids) if qa_paths else load_default_qa_items(doc_ids)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    records: list[dict[str, Any]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for item in qa_items:
            qa = item["qa"]
            run_specs = [
                {
                    "condition": "individual" if len(item["individual_document_ids"]) == 1 else f"individual:{document_id}",
                    "lora_dict": loras[document_id],
                    "adapter_document_id": document_id,
                    "composition_document_ids": None,
                    "composition_method": None,
                    "composition_scale": None,
                }
                for document_id in item["individual_document_ids"]
            ]
            run_specs.append(
                {
                    "condition": "composition",
                    "lora_dict": composed_lora,
                    "adapter_document_id": None,
                    "composition_document_ids": list(doc_ids.values()),
                    "composition_method": args.composition_method,
                    "composition_scale": args.composition_scale,
                }
            )

            for run_spec in run_specs:
                generation = shine_eval.generate_answer(
                    metamodel=metamodel,
                    tokenizer=tokenizer,
                    qwen_device=qwen_device,
                    messages=shine_eval.build_messages(qa["question"]),
                    lora_dict=run_spec["lora_dict"],
                    max_new_tokens=args.max_new_tokens,
                    conversation_max_length=args.conversation_max_length,
                )
                record = {
                    "run_id": run_id,
                    "condition": run_spec["condition"],
                    "document_id": item["document_id"],
                    "topic": item["topic"],
                    "qa_id": qa["qa_id"],
                    "question": qa["question"],
                    "expected_answer": qa["expected_answer"],
                    "answer": generation["answer"],
                    "think": generation["think"],
                    "raw_generation": generation["raw_generation"],
                    "scores": score_answer(qa["expected_answer"], generation["answer"]),
                    "qa_metadata": qa["metadata"],
                    "qa_pairs_path": item["qa_pairs_path"],
                    "required_document_ids": item["individual_document_ids"],
                    "adapter_document_id": run_spec["adapter_document_id"],
                    "composition_document_ids": run_spec["composition_document_ids"],
                    "composition_method": run_spec["composition_method"],
                    "composition_scale": run_spec["composition_scale"],
                }
                records.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                LOGGER.info("Wrote %s / %s", qa["qa_id"], run_spec["condition"])

    summary = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "docs": [str(path) for path in docs],
        "qa_paths": [str(path) for path in qa_paths] if qa_paths else None,
        "lora_paths": {
            document_id: str(lora_path_for_doc(args.lora_dir, document_id, args.lora_label))
            for document_id in doc_ids.values()
        },
        "composition_method": args.composition_method,
        "composition_scale": args.composition_scale,
        "composition_weights": {
            document_id: weight
            for document_id, weight in zip(doc_ids.values(), weights, strict=True)
        },
        "output": str(args.output),
        "summary": summarize(records),
    }
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote summary to %s", args.summary_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
