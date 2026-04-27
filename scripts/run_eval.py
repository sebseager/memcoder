#!/usr/bin/env python3
"""CLI entrypoint for the MemCoder evaluation harness.

Subcommands:

- ``predict --config <yaml>``    Run inference, write predictions.jsonl.
- ``judge   --run-dir <dir>``    Read predictions.jsonl, write judgments.jsonl.
- ``report  --run-dir <dir>``    Read judgments.jsonl, write report.md.
- ``all     --config <yaml>``    Run predict → judge → report sequentially.

Every invocation produces a fresh ``results/<run_name>_<YYYYMMDDTHHMM>/``
directory; there is no resumability.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from a checkout without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.config import load_run_config, load_snapshot  # noqa: E402
from eval.judge import run_judging  # noqa: E402
from eval.plots import render_run_plots  # noqa: E402
from eval.report import write_report  # noqa: E402
from eval.runner import run_predictions  # noqa: E402

LOGGER = logging.getLogger("memcoder.eval.cli")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _cmd_predict(args: argparse.Namespace) -> int:
    cfg = load_run_config(args.config)
    predictions = run_predictions(cfg)
    print(predictions)
    return 0


def _cmd_judge(args: argparse.Namespace) -> int:
    cfg = load_snapshot(args.run_dir)
    judgments = run_judging(cfg, args.run_dir)
    print(judgments)
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    report = write_report(args.run_dir)
    print(report)
    plot_paths = render_run_plots(args.run_dir)
    for p in plot_paths:
        print(p)
    return 0


def _cmd_all(args: argparse.Namespace) -> int:
    cfg = load_run_config(args.config)
    predictions = run_predictions(cfg)
    run_dir = predictions.parent
    # Re-load from snapshot so the judge phase reads exactly what was written.
    snapshot_cfg = load_snapshot(run_dir)
    run_judging(snapshot_cfg, run_dir)
    write_report(run_dir)
    render_run_plots(run_dir)
    print(run_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    _setup_logging()

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_predict = sub.add_parser("predict", help="Run inference and write predictions.jsonl")
    p_predict.add_argument("--config", required=True, type=Path)
    p_predict.set_defaults(func=_cmd_predict)

    p_judge = sub.add_parser("judge", help="Score predictions.jsonl with the LLM judge")
    p_judge.add_argument("--run-dir", required=True, type=Path)
    p_judge.set_defaults(func=_cmd_judge)

    p_report = sub.add_parser("report", help="Aggregate judgments.jsonl into report.md")
    p_report.add_argument("--run-dir", required=True, type=Path)
    p_report.set_defaults(func=_cmd_report)

    p_all = sub.add_parser("all", help="predict → judge → report")
    p_all.add_argument("--config", required=True, type=Path)
    p_all.set_defaults(func=_cmd_all)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
