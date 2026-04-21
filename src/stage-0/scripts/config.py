from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPOS_DIR = DATA_DIR / "repos"
OUTPUTS_DIR = ROOT / "outputs"
INSTANCES_DIR = OUTPUTS_DIR / "instances"


@dataclass(frozen=True)
class Stage0Config:
    min_stars: int = 50
    max_stars: int = 500
    min_python_files: int = 20
    target_repo_count: int = 20
    min_repo_count: int = 15
    first_commit_cutoff: date = date(2024, 11, 1)
    truncation_token_budget: int = 2048
    min_body_lines: int = 10
    max_body_lines: int = 80
    target_min_instances: int = 80
    target_max_instances: int = 120

    @property
    def active_within_days(self) -> int:
        return 60

    @property
    def pushed_since(self) -> date:
        return date.today() - timedelta(days=self.active_within_days)


DEFAULT_CONFIG = Stage0Config()


def config_as_json_dict(cfg: Stage0Config) -> dict[str, Any]:
    return {
        "min_stars": cfg.min_stars,
        "max_stars": cfg.max_stars,
        "min_python_files": cfg.min_python_files,
        "target_repo_count": cfg.target_repo_count,
        "min_repo_count": cfg.min_repo_count,
        "first_commit_cutoff": cfg.first_commit_cutoff.isoformat(),
        "truncation_token_budget": cfg.truncation_token_budget,
        "min_body_lines": cfg.min_body_lines,
        "max_body_lines": cfg.max_body_lines,
        "target_min_instances": cfg.target_min_instances,
        "target_max_instances": cfg.target_max_instances,
        "active_within_days": cfg.active_within_days,
        "pushed_since": cfg.pushed_since.isoformat(),
    }


def ensure_stage_dirs() -> None:
    for path in (DATA_DIR, REPOS_DIR, OUTPUTS_DIR, INSTANCES_DIR):
        path.mkdir(parents=True, exist_ok=True)
