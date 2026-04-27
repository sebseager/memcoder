#!/usr/bin/env python3
"""Two-checkpoint SHINE LoRA comparison: bake → predict → judge → report.

Bakes a fresh LoRA dict per (document, checkpoint) pair from each of two
SHINE hypernetwork checkpoints, runs every QA in the artifact ledger
through both LoRAs against the same Qwen base, judges the answers with
the existing OpenAI judge from ``eval/judge.py``, and emits a side-by-side
report under ``results/<run_name>_<YYYYMMDDTHHMM>/``.

LoRAs are kept in GPU memory only — no ``.pt`` files are written under
``artifacts/<repo>/loras/``, and ``ledger.json`` is never mutated. Unlike
``scripts/run_eval.py`` this script does not run ``naive`` or ``in_context``
baselines; only the LoRA-per-checkpoint case is exercised.

Multi-GPU layout: the shared Qwen base lives on ``model.qwen_cuda`` and
each checkpoint's hypernetwork is pinned to its own ``checkpoints[i].cuda``.
All three CUDA indices must be distinct.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_shine_eval as shine_eval  # noqa: E402

from eval.artifacts import iter_documents  # noqa: E402
from eval.config import (  # noqa: E402
    ArtifactSelector,
    EmbeddingConfig,
    JudgeConfig,
    ModelConfig,
    RunConfig,
)
from eval.judge import run_judging  # noqa: E402
from eval.paths import resolve_repo_path  # noqa: E402
from eval.plots import render_run_plots  # noqa: E402

LOGGER = logging.getLogger("memcoder.two_checkpoint_eval")

FAILURE_MODES = (
    "wrong_specifics",
    "missing_information",
    "off_topic",
    "refusal_or_nonresponse",
    "format_failure",
    "other",
)


@dataclass
class CheckpointSpec:
    label: str
    path: Path
    cuda: int


@dataclass
class ModelSpec:
    qwen_base: Path
    shine_root: Path
    qwen_cuda: int
    context_max_length: int
    conversation_max_length: int
    max_new_tokens: int
    seed: int


@dataclass
class TwoCkptConfig:
    run_name: str
    timestamp: str
    source_path: Path
    artifacts: list[ArtifactSelector]
    checkpoints: list[CheckpointSpec]
    model: ModelSpec
    judge: JudgeConfig
    shine_base_config: Path
    shine_overrides: list[str]
    raw: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a two-checkpoint comparison config YAML.",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Stop after writing predictions.jsonl; skip judge + report.",
    )
    return parser.parse_args()


def load_config(path: Path) -> TwoCkptConfig:
    resolved = resolve_repo_path(path)
    assert resolved is not None
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a YAML mapping: {resolved}")

    run_name = raw.get("run_name")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run_name must be a non-empty string")

    artifacts_raw = raw.get("artifacts")
    if not isinstance(artifacts_raw, list) or not artifacts_raw:
        raise ValueError("artifacts must be a non-empty list")
    artifacts: list[ArtifactSelector] = []
    for idx, entry in enumerate(artifacts_raw):
        if not isinstance(entry, dict) or "root" not in entry:
            raise ValueError(f"artifacts[{idx}] must be a mapping with 'root'")
        root = resolve_repo_path(Path(entry["root"]))
        assert root is not None
        artifacts.append(
            ArtifactSelector(
                root=root,
                difficulties=list(entry.get("difficulties") or []),
                document_ids=list(entry.get("document_ids") or []),
                topics=list(entry.get("topics") or []),
            )
        )

    ckpts_raw = raw.get("checkpoints")
    if not isinstance(ckpts_raw, list) or len(ckpts_raw) != 2:
        raise ValueError("checkpoints must be a list of exactly 2 entries")
    seen_labels: set[str] = set()
    checkpoints: list[CheckpointSpec] = []
    for entry in ckpts_raw:
        if not isinstance(entry, dict):
            raise ValueError(f"checkpoint entry is not a mapping: {entry!r}")
        label = str(entry.get("label", "")).strip()
        if not label:
            raise ValueError(f"checkpoint entry missing 'label': {entry!r}")
        if label in seen_labels:
            raise ValueError(f"duplicate checkpoint label: {label!r}")
        seen_labels.add(label)
        path_value = entry.get("path")
        if not path_value:
            raise ValueError(f"checkpoint {label!r} missing 'path'")
        ckpt_path = resolve_repo_path(Path(path_value))
        assert ckpt_path is not None
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint path does not exist: {ckpt_path}")
        if "cuda" not in entry:
            raise ValueError(
                f"checkpoint {label!r} missing 'cuda'; each hypernetwork "
                "must be pinned to its own GPU index"
            )
        checkpoints.append(
            CheckpointSpec(label=label, path=ckpt_path, cuda=int(entry["cuda"]))
        )

    model_raw = raw.get("model")
    if not isinstance(model_raw, dict):
        raise ValueError("model must be a mapping")
    qwen_base = resolve_repo_path(Path(model_raw["qwen_base"]))
    shine_root = resolve_repo_path(Path(model_raw["shine_root"]))
    assert qwen_base is not None and shine_root is not None
    model = ModelSpec(
        qwen_base=qwen_base,
        shine_root=shine_root,
        qwen_cuda=int(model_raw.get("qwen_cuda", 0)),
        context_max_length=int(model_raw.get("context_max_length", 1150)),
        conversation_max_length=int(model_raw.get("conversation_max_length", 5000)),
        max_new_tokens=int(model_raw.get("max_new_tokens", 128)),
        seed=int(model_raw.get("seed", 42)),
    )

    used_cuda = {model.qwen_cuda}
    for ckpt in checkpoints:
        if ckpt.cuda in used_cuda:
            raise ValueError(
                f"checkpoint {ckpt.label!r} requested cuda:{ckpt.cuda} which is "
                "already in use by another component"
            )
        used_cuda.add(ckpt.cuda)

    judge_raw = raw.get("judge")
    if not isinstance(judge_raw, dict):
        raise ValueError("judge must be a mapping")
    dotenv_value = judge_raw.get("dotenv_path")
    judge_prompt = resolve_repo_path(Path(judge_raw["prompt"]))
    assert judge_prompt is not None
    judge = JudgeConfig(
        provider=str(judge_raw.get("provider", "openai")),
        model=str(judge_raw["model"]),
        api_key_env=str(judge_raw.get("api_key_env", "OPENAI_API_KEY")),
        dotenv_path=resolve_repo_path(Path(dotenv_value)) if dotenv_value else None,
        concurrency=int(judge_raw.get("concurrency", 8)),
        prompt=judge_prompt,
        rubric_version=str(judge_raw.get("rubric_version", "v0")),
        taxonomy_version=str(judge_raw.get("taxonomy_version", "v0")),
        max_retries=int(judge_raw.get("max_retries", 4)),
    )

    shine_base_value = raw.get("shine_base_config", "config/shine_eval_demo.yaml")
    shine_base = resolve_repo_path(Path(shine_base_value))
    assert shine_base is not None
    if not shine_base.exists():
        raise FileNotFoundError(f"shine_base_config does not exist: {shine_base}")
    shine_overrides = list(raw.get("shine_overrides") or [])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    return TwoCkptConfig(
        run_name=run_name,
        timestamp=timestamp,
        source_path=resolved,
        artifacts=artifacts,
        checkpoints=checkpoints,
        model=model,
        judge=judge,
        shine_base_config=shine_base,
        shine_overrides=shine_overrides,
        raw=raw,
    )


def snapshot_config(cfg: TwoCkptConfig, run_dir: Path) -> Path:
    snapshot = {
        "run_name": cfg.run_name,
        "timestamp": cfg.timestamp,
        "source_path": str(cfg.source_path),
        "shine_base_config": str(cfg.shine_base_config),
        "shine_overrides": list(cfg.shine_overrides),
        "artifacts": [
            {
                "root": str(a.root),
                "difficulties": list(a.difficulties),
                "document_ids": list(a.document_ids),
                "topics": list(a.topics),
            }
            for a in cfg.artifacts
        ],
        "checkpoints": [
            {"label": c.label, "path": str(c.path), "cuda": c.cuda}
            for c in cfg.checkpoints
        ],
        "model": {
            "qwen_base": str(cfg.model.qwen_base),
            "shine_root": str(cfg.model.shine_root),
            "qwen_cuda": cfg.model.qwen_cuda,
            "context_max_length": cfg.model.context_max_length,
            "conversation_max_length": cfg.model.conversation_max_length,
            "max_new_tokens": cfg.model.max_new_tokens,
            "seed": cfg.model.seed,
        },
        "judge": {
            "provider": cfg.judge.provider,
            "model": cfg.judge.model,
            "api_key_env": cfg.judge.api_key_env,
            "dotenv_path": str(cfg.judge.dotenv_path) if cfg.judge.dotenv_path else None,
            "concurrency": cfg.judge.concurrency,
            "prompt": str(cfg.judge.prompt),
            "rubric_version": cfg.judge.rubric_version,
            "taxonomy_version": cfg.judge.taxonomy_version,
            "max_retries": cfg.judge.max_retries,
        },
    }
    out = run_dir / "run_config.yaml"
    out.write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
    return out


def build_shine_args(cfg: TwoCkptConfig) -> argparse.Namespace:
    """Synthesize the Namespace ``run_shine_eval.load_config`` expects."""
    return argparse.Namespace(
        config=cfg.shine_base_config,
        overrides=list(cfg.shine_overrides),
        design_doc=None,
        qa_pairs=None,
        output=None,
        shine_root=cfg.model.shine_root,
        checkpoint_dir=None,
        shine_checkpoint=[
            {"label": c.label, "path": str(c.path), "cuda": c.cuda}
            for c in cfg.checkpoints
        ],
        qwen_cuda=cfg.model.qwen_cuda,
        model_path=cfg.model.qwen_base,
        save_lora_dict=None,
        conditions=["shine"],
        device=None,
        context_max_length=cfg.model.context_max_length,
        conversation_max_length=cfg.model.conversation_max_length,
        max_new_tokens=cfg.model.max_new_tokens,
        seed=cfg.model.seed,
        allow_random_metanetwork=False,
    )


def stub_run_config(cfg: TwoCkptConfig) -> RunConfig:
    """Build a ``RunConfig`` adequate for ``iter_documents`` and ``run_judging``.

    Only ``artifacts`` and ``judge`` are actually read by those functions; the
    remaining fields are stubs to satisfy the dataclass contract.
    """
    return RunConfig(
        run_name=cfg.run_name,
        timestamp=cfg.timestamp,
        artifacts=list(cfg.artifacts),
        conditions=[f"shine:{c.label}" for c in cfg.checkpoints],
        routing="oracle",
        model=ModelConfig(
            qwen_base=cfg.model.qwen_base,
            shine_root=cfg.model.shine_root,
            qwen_cuda=cfg.model.qwen_cuda,
            conversation_max_length=cfg.model.conversation_max_length,
            max_new_tokens=cfg.model.max_new_tokens,
            seed=cfg.model.seed,
            lora_r=8,
            metamodel_class_path="LoraQwen.LoraQwen3ForCausalLM",
            config_class_path="LoraQwen.Qwen3Config",
        ),
        judge=cfg.judge,
        embedding=EmbeddingConfig(model=None),
        source_path=cfg.source_path,
        raw=cfg.raw,
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    cfg = load_config(args.config)

    results_root = resolve_repo_path(Path("results"))
    assert results_root is not None
    run_dir = results_root / f"{cfg.run_name}_{cfg.timestamp}"
    if run_dir.exists():
        raise FileExistsError(
            f"results directory already exists: {run_dir} "
            "(timestamp granularity is one minute; wait and re-run)"
        )
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot_config(cfg, run_dir)
    LOGGER.info("Run dir: %s", run_dir)

    eval_run_cfg = stub_run_config(cfg)
    documents = list(iter_documents(eval_run_cfg))
    if not documents:
        raise ValueError("no documents matched the artifacts selectors")
    LOGGER.info("Selected %d document(s) for comparison", len(documents))

    shine_eval.import_runtime_deps()
    shine_args = build_shine_args(cfg)
    shine_cfg = shine_eval.load_config(shine_args, require_eval_paths=False)
    shine_eval.setup_shine_imports(shine_args.shine_root)

    shine_specs, multi_mode = shine_eval.resolve_shine_specs(shine_args)
    if not multi_mode or len(shine_specs) != 2:
        raise RuntimeError(
            "expected two shine_specs in multi-checkpoint mode; got "
            f"{len(shine_specs)} (multi_mode={multi_mode})"
        )
    qwen_device = shine_eval.resolve_qwen_device(shine_args, multi_mode)
    for spec in shine_specs:
        if spec["hyper_device"] is None:
            spec["hyper_device"] = qwen_device

    if shine_eval.torch.cuda.is_available() and qwen_device.type == "cuda":
        shine_eval.torch.cuda.set_device(
            qwen_device.index
            if qwen_device.index is not None
            else shine_eval.CUDA_DEFAULT_DEVICE
        )

    LOGGER.info(
        "Multi-checkpoint mode: Qwen on %s, hypernets on %s",
        qwen_device,
        ", ".join(f"{s['label']}@{s['hyper_device']}" for s in shine_specs),
    )

    metamodel, tokenizer = shine_eval.load_base_runtime(shine_cfg, qwen_device)

    # Bake LoRAs: for each variant, attach hypernet → bake every doc → release.
    # mem_tokens are rebound to the shared metamodel before each bake so this
    # is independent of variant load order. After release the lora_dicts live
    # on qwen_device and the hypernets' memory is reclaimed.
    baked: dict[str, dict[str, Any]] = {}
    checkpoint_meta: dict[str, dict[str, Any]] = {}
    for spec in shine_specs:
        label = spec["label"]
        LOGGER.info("Attaching SHINE variant %r on %s", label, spec["hyper_device"])
        variant = shine_eval.attach_shine_variant(
            metamodel=metamodel,
            cfg=shine_cfg,
            label=label,
            checkpoint_dir=spec["path"],
            qwen_device=qwen_device,
            hyper_device=spec["hyper_device"],
            allow_random=False,
        )
        shine_eval.bind_variant_mem_tokens(metamodel, variant, qwen_device)
        baked[label] = {}
        for doc in documents:
            LOGGER.info("Baking LoRA: %s @ %s", doc.document_id, label)
            baked[label][doc.document_id] = shine_eval.build_lora_dict(
                variant["metanetwork"],
                tokenizer,
                doc.doc_text,
                variant["metalora"],
                qwen_device,
                cfg.model.context_max_length,
            )
        checkpoint_meta[label] = {
            "path": str(spec["path"]),
            "hyper_device": str(spec["hyper_device"]),
        }
        shine_eval.release_variant_runtime(variant)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "run_id": run_id,
        "run_name": cfg.run_name,
        "timestamp": cfg.timestamp,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qwen_device": str(qwen_device),
        "qwen_base": str(cfg.model.qwen_base),
        "shine_root": str(cfg.model.shine_root),
        "shine_base_config": str(cfg.shine_base_config),
        "checkpoints": [
            {
                "label": c.label,
                "path": str(c.path),
                "cuda": c.cuda,
                "hyper_device": checkpoint_meta[c.label]["hyper_device"],
            }
            for c in cfg.checkpoints
        ],
        "documents": [
            {
                "repo_id": d.repo_id,
                "document_id": d.document_id,
                "topic": d.topic,
                "difficulty": d.difficulty,
                "qa_count": len(d.qa_pairs),
            }
            for d in documents
        ],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    predictions_path = run_dir / "predictions.jsonl"
    labels = [s["label"] for s in shine_specs]
    rows_written = 0
    with predictions_path.open("w", encoding="utf-8") as out:
        for doc in documents:
            for qa in doc.qa_pairs:
                for label in labels:
                    lora_dict = baked[label][doc.document_id]
                    messages = shine_eval.build_messages(qa.question)
                    started = time.perf_counter()
                    gen = shine_eval.generate_answer(
                        metamodel=metamodel,
                        tokenizer=tokenizer,
                        qwen_device=qwen_device,
                        messages=messages,
                        lora_dict=lora_dict,
                        max_new_tokens=cfg.model.max_new_tokens,
                        conversation_max_length=cfg.model.conversation_max_length,
                    )
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    row = {
                        "run_id": run_id,
                        "repo_id": doc.repo_id,
                        "document_id": doc.document_id,
                        "topic": doc.topic,
                        "topic_slug": doc.topic_slug,
                        "difficulty": doc.difficulty,
                        "qa_id": qa.qa_id,
                        "condition": f"shine:{label}",
                        "checkpoint_label": label,
                        "shine_label": label,
                        "shine_checkpoint_dir": checkpoint_meta[label]["path"],
                        "question": qa.question,
                        "expected_answer": qa.expected_answer,
                        "answer": gen["answer"],
                        "think": gen["think"],
                        "raw_generation": gen["raw_generation"],
                        "routing": {
                            "strategy": "oracle",
                            "selected_lora_ids": [doc.document_id],
                        },
                        "lora_path": None,
                        "doc_in_context": False,
                        "timings_ms": {"generation": elapsed_ms},
                        "qa_metadata": dict(qa.metadata),
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    rows_written += 1
                    LOGGER.info(
                        "Wrote %s / %s / %s",
                        doc.document_id,
                        qa.qa_id,
                        label,
                    )

    LOGGER.info("Predictions: %d rows -> %s", rows_written, predictions_path)

    if args.predict_only:
        LOGGER.info("--predict-only set; skipping judge + report")
        return 0

    LOGGER.info("Running judge phase against %s", cfg.judge.model)
    run_judging(eval_run_cfg, run_dir)

    write_report(run_dir, labels)
    return 0


def write_report(run_dir: Path, labels: list[str]) -> Path:
    """Side-by-side report for the two checkpoints.

    Standard score / failure-mode / heatmap PNGs come from
    ``eval.plots.render_run_plots`` (which keys off the ``condition`` field
    and therefore treats each ``shine:<label>`` as its own group). A
    checkpoint-vs-checkpoint scatter is added on top because it is the most
    useful one-glance comparison plot in the two-checkpoint case.
    """
    judgments_path = run_dir / "judgments.jsonl"
    rows = [
        json.loads(line)
        for line in judgments_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no rows in {judgments_path}")

    by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    by_qa: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        label = row.get("checkpoint_label") or row.get("shine_label")
        if label not in by_label:
            continue
        by_label[label].append(row)
        key = (str(row.get("document_id")), str(row.get("qa_id")))
        by_qa[key][label] = row

    standard_plots = render_run_plots(run_dir)
    extra_plots: list[Path] = []
    scatter = _plot_score_scatter(run_dir, labels, by_qa)
    if scatter is not None:
        extra_plots.append(scatter)
    plot_paths = list(standard_plots) + extra_plots

    lines: list[str] = []
    lines.append(f"# Two-Checkpoint LoRA Comparison — {run_dir.name}")
    lines.append("")
    lines.append(f"- Total rows: **{len(rows)}**")
    lines.append(f"- Checkpoints: {', '.join(f'`{lbl}`' for lbl in labels)}")
    lines.append("")

    lines.append("## Score summary")
    lines.append("")
    lines.append("| Checkpoint | N | Mean | % score 5 | 1 | 2 | 3 | 4 | 5 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for label in labels:
        scores = [int(r.get("judge", {}).get("score", 0)) for r in by_label[label]]
        valid = [s for s in scores if 1 <= s <= 5]
        n = len(by_label[label])
        mean = (sum(valid) / len(valid)) if valid else 0.0
        pct5 = (100.0 * sum(1 for s in valid if s == 5) / len(valid)) if valid else 0.0
        hist = Counter(valid)
        lines.append(
            f"| `{label}` | {n} | {mean:.2f} | {pct5:.1f}% | "
            f"{hist.get(1, 0)} | {hist.get(2, 0)} | {hist.get(3, 0)} | "
            f"{hist.get(4, 0)} | {hist.get(5, 0)} |"
        )
    lines.append("")

    if len(labels) == 2:
        a, b = labels
        wins_a = wins_b = ties = 0
        deltas: list[int] = []
        for by_lbl in by_qa.values():
            if a not in by_lbl or b not in by_lbl:
                continue
            sa = int(by_lbl[a].get("judge", {}).get("score", 0))
            sb = int(by_lbl[b].get("judge", {}).get("score", 0))
            d = sa - sb
            deltas.append(d)
            if d > 0:
                wins_a += 1
            elif d < 0:
                wins_b += 1
            else:
                ties += 1
        lines.append("## Head-to-head")
        lines.append("")
        lines.append(f"- `{a}` wins: **{wins_a}**")
        lines.append(f"- `{b}` wins: **{wins_b}**")
        lines.append(f"- Ties: **{ties}**")
        if deltas:
            mean_delta = sum(deltas) / len(deltas)
            lines.append(
                f"- Mean score delta (`{a}` − `{b}`): **{mean_delta:+.2f}**"
            )
        lines.append("")

    lines.append("## Failure modes")
    lines.append("")
    header = "| Checkpoint | " + " | ".join(FAILURE_MODES) + " |"
    sep = "|---|" + "---|" * len(FAILURE_MODES)
    lines.append(header)
    lines.append(sep)
    for label in labels:
        tag_counts: Counter = Counter()
        for r in by_label[label]:
            tags = r.get("judge", {}).get("failure_modes") or []
            tag_counts.update(tags)
        cells = [str(tag_counts.get(m, 0)) for m in FAILURE_MODES]
        lines.append(f"| `{label}` | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Plots")
    lines.append("")
    for path in plot_paths:
        rel = path.relative_to(run_dir).as_posix()
        title = path.stem.replace("_", " ").title()
        lines.append(f"### {title}")
        lines.append("")
        lines.append(f"![{title}]({rel})")
        lines.append("")

    if len(labels) == 2:
        a, b = labels
        lines.append("## Per-question detail")
        lines.append("")
        lines.append(
            f"| Document | QA | `{a}` | `{b}` | Δ | "
            f"`{a}` failure_modes | `{b}` failure_modes |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for (doc_id, qa_id), by_lbl in sorted(by_qa.items()):
            ra = by_lbl.get(a) or {}
            rb = by_lbl.get(b) or {}
            sa = int(ra.get("judge", {}).get("score", 0))
            sb = int(rb.get("judge", {}).get("score", 0))
            delta = sa - sb
            tags_a = ", ".join(ra.get("judge", {}).get("failure_modes") or []) or "—"
            tags_b = ", ".join(rb.get("judge", {}).get("failure_modes") or []) or "—"
            lines.append(
                f"| {doc_id} | {qa_id} | {sa} | {sb} | {delta:+d} | "
                f"{tags_a} | {tags_b} |"
            )
        lines.append("")

    lines.append("## Per-question answers")
    lines.append("")
    for (doc_id, qa_id), by_lbl in sorted(by_qa.items()):
        any_row = next(iter(by_lbl.values()))
        question = any_row.get("question", "")
        expected = any_row.get("expected_answer")
        lines.append(f"### {doc_id} / {qa_id}")
        lines.append("")
        lines.append(f"**Question:** {question}")
        lines.append("")
        lines.append(f"**Expected:** {expected}")
        lines.append("")
        for label in labels:
            r = by_lbl.get(label)
            if r is None:
                continue
            j = r.get("judge", {})
            score = j.get("score")
            answer = r.get("answer", "")
            reasoning = j.get("reasoning", "")
            failure_modes = ", ".join(j.get("failure_modes") or []) or "—"
            lines.append(f"- `{label}` — score **{score}** ({failure_modes})")
            lines.append(f"  - answer: {answer}")
            if reasoning:
                lines.append(f"  - judge: {reasoning}")
        lines.append("")

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", report_path)
    return report_path


def _plot_score_scatter(
    run_dir: Path,
    labels: list[str],
    by_qa: dict[tuple[str, str], dict[str, dict[str, Any]]],
) -> Path | None:
    """Scatter checkpoint-A scores against checkpoint-B scores, jittered."""
    if len(labels) != 2:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    a, b = labels
    xs: list[int] = []
    ys: list[int] = []
    for by_lbl in by_qa.values():
        ra = by_lbl.get(a)
        rb = by_lbl.get(b)
        if ra is None or rb is None:
            continue
        sa = int(ra.get("judge", {}).get("score", 0))
        sb = int(rb.get("judge", {}).get("score", 0))
        if 1 <= sa <= 5 and 1 <= sb <= 5:
            xs.append(sa)
            ys.append(sb)
    if not xs:
        return None

    rng = np.random.default_rng(0)
    jx = np.array(xs) + rng.uniform(-0.15, 0.15, size=len(xs))
    jy = np.array(ys) + rng.uniform(-0.15, 0.15, size=len(ys))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0.5, 5.5], [0.5, 5.5], "k--", alpha=0.4, label="parity")
    ax.scatter(jx, jy, alpha=0.7)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_xticks(range(1, 6))
    ax.set_yticks(range(1, 6))
    ax.set_xlabel(f"{a} score (jittered)")
    ax.set_ylabel(f"{b} score (jittered)")
    ax.set_title("Per-question scores: scatter")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = run_dir / "score_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
