#!/usr/bin/env python3
"""Batch-generate SHINE LoRA artifacts for documents in a repo artifact ledger.

This is the artifact-level wrapper around ``generate_shine_lora.py``. It walks
``artifacts/<repo>/ledger.json``, selects document entries, writes LoRA
dictionaries under ``<difficulty>/loras/``, and updates only the LoRA fields in
the ledger so existing descriptions, QA paths, and routing examples are
preserved.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_shine_eval as shine_eval  # noqa: E402


LOGGER = logging.getLogger("memcoder.generate_artifact_loras")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        required=True,
        type=Path,
        help="Repo artifact root, e.g. artifacts/marimo-team__marimo.",
    )
    parser.add_argument(
        "--config",
        default=Path("config/shine_eval_demo.yaml"),
        type=Path,
        help="Base SHINE OmegaConf config.",
    )
    parser.add_argument(
        "--difficulty",
        action="append",
        default=[],
        help="Difficulty to include. Repeatable. Defaults to all ledger entries.",
    )
    parser.add_argument(
        "--document-id",
        action="append",
        default=[],
        help="Document ID to include. Repeatable. Defaults to all selected entries.",
    )
    parser.add_argument(
        "--topic",
        action="append",
        default=[],
        help="Topic title or topic_slug to include. Repeatable.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate LoRAs even when the expected output files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected docs and intended outputs without loading models.",
    )
    parser.add_argument("--shine-root", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument(
        "--shine-checkpoint",
        action="append",
        default=[],
        metavar="label=NAME,path=DIR[,cuda=N]",
        help="Repeatable multi-checkpoint specification, matching run_shine_eval.py.",
    )
    parser.add_argument("--qwen-cuda", type=int, default=None)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--context-max-length", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--allow-random-metanetwork",
        action="store_true",
        help="Allow generation without a checkpoint. Intended only for smoke tests.",
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


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object at {path}")
    return data


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def selected_entries(
    ledger: dict[str, Any],
    *,
    difficulties: set[str],
    document_ids: set[str],
    topics: set[str],
) -> list[tuple[str, dict[str, Any]]]:
    documents = ledger.get("documents")
    if not isinstance(documents, dict):
        raise ValueError("ledger.json missing documents map")

    selected: list[tuple[str, dict[str, Any]]] = []
    for document_id, entry in documents.items():
        if not isinstance(entry, dict):
            continue
        difficulty = str(entry.get("difficulty") or "")
        topic = str(entry.get("topic") or "")
        topic_slug = str(entry.get("topic_slug") or "")
        if difficulties and difficulty not in difficulties:
            continue
        if document_ids and document_id not in document_ids:
            continue
        if topics and topic not in topics and topic_slug not in topics:
            continue
        selected.append((str(document_id), entry))
    return selected


def expected_lora_paths(
    artifact_root: Path,
    document_id: str,
    entry: dict[str, Any],
    shine_specs: list[dict[str, Any]],
    multi_mode: bool,
) -> list[tuple[str, Path]]:
    difficulty = str(entry.get("difficulty") or "")
    if not difficulty:
        raise ValueError(f"{document_id}: ledger entry missing difficulty")
    base = artifact_root / difficulty / "loras" / f"{document_id}.pt"
    if multi_mode:
        return [
            (str(spec["label"]), base.with_name(f"{base.stem}__{spec['label']}{base.suffix}"))
            for spec in shine_specs
        ]
    return [(str(shine_specs[0]["label"]), base)]


def all_outputs_exist(paths: list[tuple[str, Path]]) -> bool:
    return all(path.exists() for _label, path in paths)


def update_lora_fields(
    ledger: dict[str, Any],
    artifact_root: Path,
    document_id: str,
    saved_paths: list[tuple[str, Path]],
    multi_mode: bool,
) -> None:
    entry = ledger["documents"][document_id]
    files = entry.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"{document_id}: ledger entry missing files map")

    def rel(path: Path) -> str:
        return path.resolve().relative_to(artifact_root.resolve()).as_posix()

    if multi_mode:
        files["lora"] = None
        files["lora_variants"] = {label: rel(path) for label, path in saved_paths}
    else:
        files["lora"] = rel(saved_paths[0][1])
        files.pop("lora_variants", None)


def configure_shine(args: argparse.Namespace, first_doc: Path) -> tuple[Any, list[dict[str, Any]], bool, Any, Any, Any]:
    args.design_doc = first_doc
    args.qa_pairs = None
    args.output = None
    args.save_lora_dict = None
    args.conditions = ["shine"]
    args.conversation_max_length = None
    args.max_new_tokens = None

    shine_eval.import_runtime_deps()
    cfg = shine_eval.load_config(args, require_eval_paths=False)
    shine_eval.setup_shine_imports(args.shine_root)

    shine_specs, multi_mode = shine_eval.resolve_shine_specs(args)
    qwen_device = shine_eval.resolve_qwen_device(args, multi_mode)
    for spec in shine_specs:
        if spec["hyper_device"] is None:
            spec["hyper_device"] = qwen_device

    if shine_eval.torch.cuda.is_available() and qwen_device.type == "cuda":
        shine_eval.torch.cuda.set_device(
            qwen_device.index if qwen_device.index is not None else shine_eval.CUDA_DEFAULT_DEVICE
        )

    metamodel, tokenizer = shine_eval.load_base_runtime(cfg, qwen_device)
    return cfg, shine_specs, multi_mode, qwen_device, metamodel, tokenizer


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    artifact_root = resolve_repo_path(args.artifact_root)
    ledger_path = artifact_root / "ledger.json"
    ledger = load_json(ledger_path)

    entries = selected_entries(
        ledger,
        difficulties=set(args.difficulty),
        document_ids=set(args.document_id),
        topics=set(args.topic),
    )
    if not entries:
        raise SystemExit("No ledger entries matched the requested filters.")

    doc_paths: dict[str, Path] = {}
    for document_id, entry in entries:
        files = entry.get("files")
        if not isinstance(files, dict) or not isinstance(files.get("doc"), str):
            raise ValueError(f"{document_id}: ledger entry missing files.doc")
        doc_path = artifact_root / files["doc"]
        if not doc_path.exists():
            raise FileNotFoundError(f"{document_id}: design doc not found: {doc_path}")
        doc_paths[document_id] = doc_path

    first_doc = next(iter(doc_paths.values()))
    if args.dry_run:
        # Parse specs without loading torch/model dependencies.
        args.design_doc = first_doc
        args.qa_pairs = None
        args.output = None
        args.save_lora_dict = None
        args.conditions = ["shine"]
        args.conversation_max_length = None
        args.max_new_tokens = None
        shine_eval.OmegaConf = None
        print(f"Would generate LoRAs for {len(entries)} document(s) under {artifact_root}")
        for document_id, entry in entries:
            difficulty = entry.get("difficulty")
            print(f"{document_id}\t{difficulty}\t{artifact_root / difficulty / 'loras' / (document_id + '.pt')}")
        return 0

    _cfg, shine_specs, multi_mode, qwen_device, metamodel, tokenizer = configure_shine(args, first_doc)

    generated = 0
    skipped = 0
    for document_id, entry in entries:
        outputs = expected_lora_paths(artifact_root, document_id, entry, shine_specs, multi_mode)
        if not args.force and all_outputs_exist(outputs):
            LOGGER.info("Skipping %s: LoRA output(s) already exist", document_id)
            update_lora_fields(ledger, artifact_root, document_id, outputs, multi_mode)
            skipped += 1
            continue

        document, doc_metadata = shine_eval.load_design_doc(doc_paths[document_id])
        metadata_id = str(doc_metadata.get("document_id") or document_id)
        if metadata_id != document_id:
            raise ValueError(
                f"{document_id}: design doc document_id mismatch: {metadata_id}"
            )

        saved_paths: list[tuple[str, Path]] = []
        for spec in shine_specs:
            variant = shine_eval.attach_shine_variant(
                metamodel=metamodel,
                cfg=_cfg,
                label=spec["label"],
                checkpoint_dir=spec["path"],
                qwen_device=qwen_device,
                hyper_device=spec["hyper_device"],
                allow_random=args.allow_random_metanetwork,
            )
            shine_eval.bind_variant_mem_tokens(metamodel, variant, qwen_device)
            LOGGER.info("Generating LoRA dictionary for %s (%s)", document_id, spec["label"])
            lora_dict = shine_eval.build_lora_dict(
                variant["metanetwork"],
                tokenizer,
                document,
                variant["metalora"],
                qwen_device,
                args.context_max_length,
            )
            base_path = artifact_root / str(entry["difficulty"]) / "loras" / f"{document_id}.pt"
            saved_path = shine_eval._save_lora_dict(
                base_path,
                spec["label"],
                lora_dict,
                multi=multi_mode,
            )
            saved_paths.append((str(spec["label"]), saved_path))
            shine_eval.release_variant_runtime(variant)

        update_lora_fields(ledger, artifact_root, document_id, saved_paths, multi_mode)
        save_json(ledger_path, ledger)
        generated += 1

    save_json(ledger_path, ledger)
    LOGGER.info(
        "LoRA generation complete: generated=%d skipped=%d ledger=%s",
        generated,
        skipped,
        ledger_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
