#!/usr/bin/env python3
"""Generate and save SHINE LoRA dictionary artifacts for one design document."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import run_shine_eval as shine_eval


LOGGER = logging.getLogger("memcoder.generate_shine_lora")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LORA_DIR = Path("artifacts/antirez__kilo/easy/loras")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=Path("config/shine_eval_demo.yaml"),
        type=Path,
        help="MemCoder SHINE eval config file.",
    )
    parser.add_argument(
        "--design-doc",
        type=Path,
        help="Design document JSON or text file. Defaults to the config's design_doc.",
    )
    parser.add_argument(
        "--qa-pairs",
        type=Path,
        help="Optional QA path to record in the artifact ledger. Defaults to sibling qas/<document_id>.json when present.",
    )
    parser.add_argument(
        "--lora-dir",
        type=Path,
        default=DEFAULT_LORA_DIR,
        help="Directory for generated LoRA artifacts when --output is not supplied.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .pt path. In multi-checkpoint mode, labels are appended to the stem.",
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
    parser.add_argument(
        "--no-update-ledger",
        action="store_true",
        help="Do not refresh artifacts/<repo>/ledger.json after saving.",
    )
    return parser.parse_args()


def default_qa_path(design_doc_path: Path, document_id: str) -> Path | None:
    if design_doc_path.parent.name != "docs":
        return None
    candidate = design_doc_path.parent.parent / "qas" / f"{document_id}.json"
    return candidate if candidate.exists() else None


def default_lora_output(lora_dir: Path, document_id: str) -> Path:
    lora_dir = shine_eval.resolve_repo_path(lora_dir)
    return lora_dir / f"{document_id}.pt"


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    requested_lora_output = args.output
    requested_qa_pairs = args.qa_pairs
    args.conditions = ["shine"]
    args.qa_pairs = None
    args.output = None
    args.conversation_max_length = None
    args.max_new_tokens = None
    args.save_lora_dict = None

    shine_eval.import_runtime_deps()
    cfg = shine_eval.load_config(args, require_eval_paths=False)
    shine_eval.setup_shine_imports(args.shine_root)

    document, doc_metadata = shine_eval.load_design_doc(args.design_doc)
    document_id = str(doc_metadata.get("document_id") or args.design_doc.stem)
    args.save_lora_dict = shine_eval.resolve_repo_path(requested_lora_output) if requested_lora_output else default_lora_output(args.lora_dir, document_id)
    qa_pairs_path = shine_eval.resolve_repo_path(requested_qa_pairs) if requested_qa_pairs else default_qa_path(args.design_doc, document_id)

    shine_specs, multi_mode = shine_eval.resolve_shine_specs(args)
    qwen_device = shine_eval.resolve_qwen_device(args, multi_mode)
    for spec in shine_specs:
        if spec["hyper_device"] is None:
            spec["hyper_device"] = qwen_device

    if shine_eval.torch.cuda.is_available() and qwen_device.type == "cuda":
        shine_eval.torch.cuda.set_device(
            qwen_device.index if qwen_device.index is not None else shine_eval.CUDA_DEFAULT_DEVICE
        )

    LOGGER.info("Loaded design doc %s", args.design_doc)
    metamodel, tokenizer = shine_eval.load_base_runtime(cfg, qwen_device)

    saved_lora_paths: list[tuple[str, Path]] = []
    for spec in shine_specs:
        variant = shine_eval.attach_shine_variant(
            metamodel=metamodel,
            cfg=cfg,
            label=spec["label"],
            checkpoint_dir=spec["path"],
            qwen_device=qwen_device,
            hyper_device=spec["hyper_device"],
            allow_random=args.allow_random_metanetwork,
        )
        shine_eval.bind_variant_mem_tokens(metamodel, variant, qwen_device)
        LOGGER.info("Generating LoRA dictionary for %s from %s", spec["label"], document_id)
        lora_dict = shine_eval.build_lora_dict(
            variant["metanetwork"],
            tokenizer,
            document,
            variant["metalora"],
            qwen_device,
            args.context_max_length,
        )
        saved_path = shine_eval._save_lora_dict(
            args.save_lora_dict,
            spec["label"],
            lora_dict,
            multi=multi_mode,
        )
        saved_lora_paths.append((spec["label"], saved_path))
        shine_eval.release_variant_runtime(variant)

    if not args.no_update_ledger and qa_pairs_path is not None:
        shine_eval.update_ledger(
            design_doc_path=args.design_doc,
            qa_pairs_path=qa_pairs_path,
            doc_metadata=doc_metadata,
            saved_lora_paths=saved_lora_paths,
        )

    for label, path in saved_lora_paths:
        print(f"{label}\t{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
