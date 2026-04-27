#!/usr/bin/env python3
"""Run LoRA composition eval scopes for the fake three-document fixture."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path("artifacts/fake_lora_composition/easy")
DOCS = {
    "fake_alpha_profile": ARTIFACT_DIR / "docs" / "fake_alpha_profile.json",
    "fake_beta_profile": ARTIFACT_DIR / "docs" / "fake_beta_profile.json",
    "fake_gamma_profile": ARTIFACT_DIR / "docs" / "fake_gamma_profile.json",
}
SCOPES = [
    ("individual/fake_alpha_profile", ["fake_alpha_profile"], ARTIFACT_DIR / "qas" / "fake_alpha_profile.json"),
    ("individual/fake_beta_profile", ["fake_beta_profile"], ARTIFACT_DIR / "qas" / "fake_beta_profile.json"),
    ("individual/fake_gamma_profile", ["fake_gamma_profile"], ARTIFACT_DIR / "qas" / "fake_gamma_profile.json"),
    (
        "needs_two/alpha_beta",
        ["fake_alpha_profile", "fake_beta_profile"],
        ARTIFACT_DIR / "qas" / "needs_two_alpha_beta.json",
    ),
    (
        "needs_two/alpha_gamma",
        ["fake_alpha_profile", "fake_gamma_profile"],
        ARTIFACT_DIR / "qas" / "needs_two_alpha_gamma.json",
    ),
    (
        "needs_two/beta_gamma",
        ["fake_beta_profile", "fake_gamma_profile"],
        ARTIFACT_DIR / "qas" / "needs_two_beta_gamma.json",
    ),
    (
        "needs_all_three/all_three",
        ["fake_alpha_profile", "fake_beta_profile", "fake_gamma_profile"],
        ARTIFACT_DIR / "qas" / "needs_all_three.json",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=Path("config/shine_eval_demo.yaml"), type=Path)
    parser.add_argument("--artifact-dir", default=ARTIFACT_DIR, type=Path)
    parser.add_argument("--lora-dir", default=ARTIFACT_DIR / "loras", type=Path)
    parser.add_argument("--output-dir", default=ARTIFACT_DIR / "results", type=Path)
    parser.add_argument("--composition-method", choices=["rank_average", "rank_sum"], default="rank_average")
    parser.add_argument("--composition-scale", type=float, default=1.0)
    parser.add_argument("--skip-generate-loras", action="store_true")
    parser.add_argument("--force-loras", action="store_true")
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
    parser.add_argument("--shine-checkpoint", action="append", default=[])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def add_optional(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def main() -> int:
    args = parse_args()
    artifact_dir = args.artifact_dir
    docs = {doc_id: artifact_dir / "docs" / f"{doc_id}.json" for doc_id in DOCS}

    for scope_name, doc_ids, qa_path in SCOPES:
        output_path = args.output_dir / scope_name / "lora_composition_results.jsonl"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_lora_composition_eval.py"),
            "--config",
            str(args.config),
            "--lora-dir",
            str(args.lora_dir),
            "--qa-pairs",
            str(artifact_dir / qa_path.relative_to(ARTIFACT_DIR)),
            "--output",
            str(output_path),
            "--composition-method",
            args.composition_method,
            "--composition-scale",
            str(args.composition_scale),
        ]
        for doc_id in doc_ids:
            cmd.extend(["--doc", str(docs[doc_id])])
        if args.skip_generate_loras:
            cmd.append("--skip-generate-loras")
        if args.force_loras:
            cmd.append("--force-loras")
        if args.allow_random_metanetwork:
            cmd.append("--allow-random-metanetwork")
        add_optional(cmd, "--shine-root", args.shine_root)
        add_optional(cmd, "--checkpoint-dir", args.checkpoint_dir)
        add_optional(cmd, "--qwen-cuda", args.qwen_cuda)
        add_optional(cmd, "--model-path", args.model_path)
        add_optional(cmd, "--device", args.device)
        add_optional(cmd, "--context-max-length", args.context_max_length)
        add_optional(cmd, "--conversation-max-length", args.conversation_max_length)
        add_optional(cmd, "--max-new-tokens", args.max_new_tokens)
        add_optional(cmd, "--seed", args.seed)
        for spec in args.shine_checkpoint:
            cmd.extend(["--shine-checkpoint", spec])
        for override in args.overrides:
            cmd.extend(["--set", override])

        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
