"""Predictions phase of the eval harness.

For every selected ledger entry, run all configured conditions and write one
JSONL row per ``(doc, qa, condition)`` to ``predictions.jsonl`` under the
results directory. The output is a fresh log every invocation — there is no
resumability or row-skipping.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import composition, model as model_module
from .artifacts import DocumentRecord, QAPair, iter_documents
from .config import RunConfig
from .routing import RoutingDecision, make_router

LOGGER = logging.getLogger("memcoder.eval.runner")


def run_predictions(cfg: RunConfig) -> Path:
    """Execute the predictions phase. Returns the path to ``predictions.jsonl``."""
    run_dir = cfg.results_dir()
    if run_dir.exists():
        raise FileExistsError(
            f"results directory already exists: {run_dir} "
            "(timestamps are minute-grained; wait a minute and re-run)"
        )
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg.snapshot(run_dir)

    documents = list(iter_documents(cfg))
    LOGGER.info("Selected %d document(s) for evaluation", len(documents))
    _validate_documents_for_conditions(documents, cfg.conditions, cfg.routing)

    model_module.ensure_runtime()
    model_module.setup_shine_imports(cfg.model.shine_root)
    qwen_device = model_module.resolve_qwen_device(cfg.model.qwen_cuda)
    metamodel, tokenizer = model_module.load_qwen_runtime(cfg.model, qwen_device)
    lora_dtype = model_module.preferred_model_dtype(qwen_device)
    router = make_router(
        cfg.routing,
        routing_results_paths=cfg.embedding.routing_results,
        top_k=cfg.embedding.top_k,
    )

    run_id = _make_run_id()
    manifest = _build_manifest(cfg, run_id, qwen_device, documents)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    predictions_path = run_dir / "predictions.jsonl"
    rows_written = 0
    with predictions_path.open("w", encoding="utf-8") as out:
        for doc in documents:
            current_cache_key: tuple[str, ...] | None = None
            current_lora_dict = None
            try:
                for qa in doc.qa_pairs:
                    shine_decision: RoutingDecision | None = None
                    if "shine" in cfg.conditions:
                        shine_decision = router.select(doc, qa)
                        if shine_decision is None:
                            LOGGER.warning(
                                "Skipping shine for %s/%s: router %r returned no decision",
                                doc.document_id,
                                qa.qa_id,
                                cfg.routing,
                            )
                        else:
                            cache_key = tuple(
                                sorted(str(p) for p in shine_decision.lora_paths)
                            )
                            if cache_key != current_cache_key:
                                loaded = [
                                    model_module.load_lora_dict(p, qwen_device, dtype=lora_dtype)
                                    for p in shine_decision.lora_paths
                                ]
                                new_composed = composition.compose_top_k(loaded)
                                # Drop the per-LoRA dicts; only the composed dict is
                                # needed downstream. ``compose_top_k`` short-circuits
                                # to identity at len==1, so skip the free in that case.
                                for d in loaded:
                                    if d is not new_composed:
                                        model_module.free_lora_dict(d)
                                del loaded
                                if current_lora_dict is not None:
                                    model_module.free_lora_dict(current_lora_dict)
                                current_lora_dict = new_composed
                                current_cache_key = cache_key

                    for cond in cfg.conditions:
                        if cond == "shine" and (
                            shine_decision is None or current_lora_dict is None
                        ):
                            continue
                        row = _run_one(
                            run_id=run_id,
                            cfg=cfg,
                            doc=doc,
                            qa=qa,
                            condition=cond,
                            metamodel=metamodel,
                            tokenizer=tokenizer,
                            qwen_device=qwen_device,
                            lora_dict=current_lora_dict if cond == "shine" else None,
                            shine_decision=shine_decision if cond == "shine" else None,
                        )
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out.flush()
                        rows_written += 1
                        LOGGER.info(
                            "Wrote %s / %s / %s",
                            doc.document_id,
                            qa.qa_id,
                            cond,
                        )
            finally:
                if current_lora_dict is not None:
                    model_module.free_lora_dict(current_lora_dict)
                    current_lora_dict = None
                    current_cache_key = None

    LOGGER.info("Predictions complete: %d rows -> %s", rows_written, predictions_path)
    return predictions_path


def _validate_documents_for_conditions(
    documents: list[DocumentRecord], conditions: list[str], routing: str
) -> None:
    if not documents:
        raise ValueError("no documents matched the artifacts selectors")
    # Only oracle gates on per-doc files.lora; embedding routing can serve a
    # qa from a LoRA baked from any other doc.
    if "shine" in conditions and routing == "oracle":
        with_lora = sum(1 for d in documents if d.lora_path is not None)
        if with_lora == 0:
            LOGGER.warning(
                "conditions include 'shine' but no selected ledger entry has a "
                "pre-baked files.lora; the shine condition will be skipped for "
                "every document — only naive/in_context rows will be written"
            )
        elif with_lora < len(documents):
            LOGGER.warning(
                "shine condition will be skipped for %d/%d document(s) lacking a "
                "pre-baked LoRA",
                len(documents) - with_lora,
                len(documents),
            )


def _run_one(
    *,
    run_id: str,
    cfg: RunConfig,
    doc: DocumentRecord,
    qa: QAPair,
    condition: str,
    metamodel: Any,
    tokenizer: Any,
    qwen_device: Any,
    lora_dict: Any,
    shine_decision: RoutingDecision | None,
) -> dict[str, Any]:
    doc_for_prompt = doc.doc_text if condition == "in_context" else None
    messages = model_module.build_messages(
        qa.question, doc_for_prompt, condition=condition
    )

    started = time.perf_counter()
    generation = model_module.generate_answer(
        metamodel=metamodel,
        tokenizer=tokenizer,
        qwen_device=qwen_device,
        messages=messages,
        lora_dict=lora_dict,
        max_new_tokens=cfg.model.max_new_tokens,
        conversation_max_length=cfg.model.conversation_max_length,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    routing_field: dict[str, Any]
    lora_path_str: str | None
    lora_paths_list: list[str] | None
    if condition == "shine" and shine_decision is not None:
        routing_field = {
            "strategy": shine_decision.strategy,
            "selected_lora_ids": list(shine_decision.selected_lora_ids),
        }
        lora_paths_list = [str(p) for p in shine_decision.lora_paths]
        # Top-1 mirrored at ``lora_path`` for back-compat with judge.py /
        # report.py and any external consumers reading the legacy field.
        lora_path_str = lora_paths_list[0] if lora_paths_list else None
    else:
        routing_field = {"strategy": "n/a", "selected_lora_ids": []}
        lora_path_str = None
        lora_paths_list = None

    return {
        "run_id": run_id,
        "repo_id": doc.repo_id,
        "document_id": doc.document_id,
        "topic": doc.topic,
        "topic_slug": doc.topic_slug,
        "difficulty": doc.difficulty,
        "qa_id": qa.qa_id,
        "condition": condition,
        "shine_label": "shine" if condition == "shine" else None,
        "shine_checkpoint_dir": None,
        "question": qa.question,
        "expected_answer": qa.expected_answer,
        "answer": generation["answer"],
        "think": generation["think"],
        "raw_generation": generation["raw_generation"],
        "routing": routing_field,
        "lora_path": lora_path_str,
        "lora_paths": lora_paths_list,
        "doc_in_context": condition == "in_context",
        "timings_ms": {"generation": elapsed_ms},
        "qa_metadata": dict(qa.metadata),
    }


def _make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[1]),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _build_manifest(
    cfg: RunConfig,
    run_id: str,
    qwen_device: Any,
    documents: list[DocumentRecord],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_name": cfg.run_name,
        "timestamp": cfg.timestamp,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "qwen_device": str(qwen_device),
        "qwen_base": str(cfg.model.qwen_base),
        "shine_root": str(cfg.model.shine_root),
        "conditions": list(cfg.conditions),
        "routing": cfg.routing,
        "documents": [
            {
                "repo_id": d.repo_id,
                "document_id": d.document_id,
                "topic": d.topic,
                "difficulty": d.difficulty,
                "description": d.description,
                "lora_path": str(d.lora_path) if d.lora_path else None,
                "qa_count": len(d.qa_pairs),
            }
            for d in documents
        ],
    }
