"""Run-config loader for the MemCoder evaluation harness.

The run config is a self-contained YAML — it carries every model, judge, and
artifact path the harness needs without depending on the legacy
``shine_eval_demo.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .paths import resolve_repo_path

VALID_CONDITIONS = ("naive", "in_context", "shine")
VALID_ROUTING = ("oracle", "embedding")


@dataclass
class ArtifactSelector:
    root: Path
    difficulties: list[str] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    qwen_base: Path
    shine_root: Path
    qwen_cuda: int
    conversation_max_length: int
    max_new_tokens: int
    seed: int
    lora_r: int
    metamodel_class_path: str
    config_class_path: str


@dataclass
class JudgeConfig:
    provider: str
    model: str
    api_key_env: str
    dotenv_path: Path | None
    concurrency: int
    prompt: Path
    rubric_version: str
    taxonomy_version: str
    max_retries: int


@dataclass
class EmbeddingConfig:
    """Placeholder; populated when embedding routing is wired up."""

    model: str | None = None


@dataclass
class RunConfig:
    run_name: str
    timestamp: str
    artifacts: list[ArtifactSelector]
    conditions: list[str]
    routing: str
    model: ModelConfig
    judge: JudgeConfig
    embedding: EmbeddingConfig
    source_path: Path
    raw: dict[str, Any]

    def results_dir(self) -> Path:
        """``<repo>/results/<run_name>_<YYYYMMDDTHHMM>/``."""
        root = resolve_repo_path(Path("results"))
        assert root is not None
        return root / f"{self.run_name}_{self.timestamp}"

    def snapshot(self, run_dir: Path) -> Path:
        """Write a post-resolution copy of the config under ``run_dir``."""
        out = run_dir / "run_config.yaml"
        snapshot = _to_snapshot(self)
        out.write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
        return out


def _require(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise ValueError(f"missing required key {key!r} in {where}")
    return d[key]


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    raise ValueError(f"expected list, got {type(value).__name__}")


def _make_timestamp() -> str:
    """UTC timestamp at minute granularity, matching ``results/<run>_<ts>/``."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")


def load_run_config(path: Path) -> RunConfig:
    path = resolve_repo_path(path)
    assert path is not None
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"run config must be a YAML mapping: {path}")

    run_name = _require(raw, "run_name", "run config")
    if not isinstance(run_name, str) or not run_name.strip():
        raise ValueError("run_name must be a non-empty string")

    artifacts_raw = _require(raw, "artifacts", "run config")
    if not isinstance(artifacts_raw, list) or not artifacts_raw:
        raise ValueError("artifacts must be a non-empty list")
    artifacts = [_parse_artifact(entry, idx) for idx, entry in enumerate(artifacts_raw)]

    conditions = _as_list(raw.get("conditions"))
    if not conditions:
        raise ValueError("conditions must be a non-empty list")
    for cond in conditions:
        if cond not in VALID_CONDITIONS:
            raise ValueError(
                f"invalid condition {cond!r}; expected one of {VALID_CONDITIONS}"
            )

    routing = raw.get("routing", "oracle")
    if routing not in VALID_ROUTING:
        raise ValueError(f"routing must be one of {VALID_ROUTING}, got {routing!r}")

    model = _parse_model(_require(raw, "model", "run config"))
    judge = _parse_judge(_require(raw, "judge", "run config"))
    embedding = _parse_embedding(raw.get("embedding") or {})

    return RunConfig(
        run_name=run_name,
        timestamp=_make_timestamp(),
        artifacts=artifacts,
        conditions=list(conditions),
        routing=routing,
        model=model,
        judge=judge,
        embedding=embedding,
        source_path=path,
        raw=raw,
    )


def _parse_artifact(entry: Any, idx: int) -> ArtifactSelector:
    if not isinstance(entry, dict):
        raise ValueError(f"artifacts[{idx}] must be a mapping")
    root = resolve_repo_path(Path(_require(entry, "root", f"artifacts[{idx}]")))
    assert root is not None
    return ArtifactSelector(
        root=root,
        difficulties=list(_as_list(entry.get("difficulties"))),
        document_ids=list(_as_list(entry.get("document_ids"))),
        topics=list(_as_list(entry.get("topics"))),
    )


def _parse_model(raw: Any) -> ModelConfig:
    if not isinstance(raw, dict):
        raise ValueError("model must be a mapping")
    qwen_base = resolve_repo_path(Path(_require(raw, "qwen_base", "model")))
    shine_root = resolve_repo_path(Path(_require(raw, "shine_root", "model")))
    assert qwen_base is not None and shine_root is not None
    return ModelConfig(
        qwen_base=qwen_base,
        shine_root=shine_root,
        qwen_cuda=int(raw.get("qwen_cuda", 0)),
        conversation_max_length=int(_require(raw, "conversation_max_length", "model")),
        max_new_tokens=int(_require(raw, "max_new_tokens", "model")),
        seed=int(raw.get("seed", 42)),
        lora_r=int(_require(raw, "lora_r", "model")),
        metamodel_class_path=str(
            raw.get("metamodel_class_path", "LoraQwen.LoraQwen3ForCausalLM")
        ),
        config_class_path=str(raw.get("config_class_path", "LoraQwen.Qwen3Config")),
    )


def _parse_judge(raw: Any) -> JudgeConfig:
    if not isinstance(raw, dict):
        raise ValueError("judge must be a mapping")
    prompt = resolve_repo_path(Path(_require(raw, "prompt", "judge")))
    assert prompt is not None
    dotenv_value = raw.get("dotenv_path")
    dotenv_path = resolve_repo_path(Path(dotenv_value)) if dotenv_value else None
    return JudgeConfig(
        provider=str(raw.get("provider", "openai")),
        model=str(_require(raw, "model", "judge")),
        api_key_env=str(raw.get("api_key_env", "OPENAI_API_KEY")),
        dotenv_path=dotenv_path,
        concurrency=int(raw.get("concurrency", 8)),
        prompt=prompt,
        rubric_version=str(raw.get("rubric_version", "v0")),
        taxonomy_version=str(raw.get("taxonomy_version", "v0")),
        max_retries=int(raw.get("max_retries", 4)),
    )


def _parse_embedding(raw: Any) -> EmbeddingConfig:
    if not isinstance(raw, dict):
        raise ValueError("embedding must be a mapping")
    model = raw.get("model")
    return EmbeddingConfig(model=str(model) if model else None)


def _to_snapshot(cfg: RunConfig) -> dict[str, Any]:
    """Serializable view of the resolved config (paths as strings)."""
    return {
        "run_name": cfg.run_name,
        "timestamp": cfg.timestamp,
        "source_path": str(cfg.source_path),
        "artifacts": [
            {
                "root": str(a.root),
                "difficulties": list(a.difficulties),
                "document_ids": list(a.document_ids),
                "topics": list(a.topics),
            }
            for a in cfg.artifacts
        ],
        "conditions": list(cfg.conditions),
        "routing": cfg.routing,
        "model": {
            "qwen_base": str(cfg.model.qwen_base),
            "shine_root": str(cfg.model.shine_root),
            "qwen_cuda": cfg.model.qwen_cuda,
            "conversation_max_length": cfg.model.conversation_max_length,
            "max_new_tokens": cfg.model.max_new_tokens,
            "seed": cfg.model.seed,
            "lora_r": cfg.model.lora_r,
            "metamodel_class_path": cfg.model.metamodel_class_path,
            "config_class_path": cfg.model.config_class_path,
        },
        "embedding": {"model": cfg.embedding.model},
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


def load_snapshot(run_dir: Path) -> RunConfig:
    """Reload a snapshot written by ``RunConfig.snapshot``.

    Used by ``judge`` and ``report`` subcommands so they don't require the
    user to re-pass the original config.
    """
    snapshot_path = run_dir / "run_config.yaml"
    raw = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"snapshot must be a mapping: {snapshot_path}")

    artifacts = [
        ArtifactSelector(
            root=Path(a["root"]),
            difficulties=list(a.get("difficulties") or []),
            document_ids=list(a.get("document_ids") or []),
            topics=list(a.get("topics") or []),
        )
        for a in raw["artifacts"]
    ]
    model_raw = raw["model"]
    model = ModelConfig(
        qwen_base=Path(model_raw["qwen_base"]),
        shine_root=Path(model_raw["shine_root"]),
        qwen_cuda=int(model_raw["qwen_cuda"]),
        conversation_max_length=int(model_raw["conversation_max_length"]),
        max_new_tokens=int(model_raw["max_new_tokens"]),
        seed=int(model_raw["seed"]),
        lora_r=int(model_raw["lora_r"]),
        metamodel_class_path=str(model_raw["metamodel_class_path"]),
        config_class_path=str(model_raw["config_class_path"]),
    )
    judge_raw = raw["judge"]
    judge = JudgeConfig(
        provider=str(judge_raw["provider"]),
        model=str(judge_raw["model"]),
        api_key_env=str(judge_raw["api_key_env"]),
        dotenv_path=Path(judge_raw["dotenv_path"]) if judge_raw.get("dotenv_path") else None,
        concurrency=int(judge_raw["concurrency"]),
        prompt=Path(judge_raw["prompt"]),
        rubric_version=str(judge_raw["rubric_version"]),
        taxonomy_version=str(judge_raw["taxonomy_version"]),
        max_retries=int(judge_raw["max_retries"]),
    )
    embedding = EmbeddingConfig(
        model=str(raw["embedding"]["model"]) if raw["embedding"].get("model") else None
    )
    return RunConfig(
        run_name=str(raw["run_name"]),
        timestamp=str(raw["timestamp"]),
        artifacts=artifacts,
        conditions=list(raw["conditions"]),
        routing=str(raw["routing"]),
        model=model,
        judge=judge,
        embedding=embedding,
        source_path=Path(raw["source_path"]),
        raw=raw,
    )
