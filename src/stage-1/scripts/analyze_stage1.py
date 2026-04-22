from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import BOOTSTRAP_SAMPLES, MODEL_ID, SEED, get_stage1_paths
from helpers import load_json


def bootstrap_ratio(
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float | None, list[float]]:
    rng = np.random.default_rng(seed)
    n = len(b)
    if n == 0:
        return None, []
    ratios = []
    for _ in range(samples):
        idx = rng.integers(0, n, size=n)
        b_m = float(np.mean(b[idx]))
        c_m = float(np.mean(c[idx]))
        d_m = float(np.mean(d[idx]))
        den = c_m - b_m
        if abs(den) < 1e-12:
            continue
        ratios.append((d_m - b_m) / den)
    if not ratios:
        return None, []
    return float(np.mean(ratios)), ratios


def load_eval_csv(path: Path) -> pd.DataFrame:
    cols = [
        "instance_id",
        "pass_at_1_proxy",
        "bleu4",
        "gap_stratum",
        "bleu_gap_bc",
    ]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=cols)

    for col in cols:
        if col not in df.columns:
            df[col] = np.nan if col == "bleu_gap_bc" else "unknown"
    return df[cols]


def safe_mean(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def ci95(vals: list[float]) -> list[float] | None:
    if not vals:
        return None
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def summarize_recovery(
    df: pd.DataFrame,
    metric: str,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    b = df[f"{metric}_B"].to_numpy(dtype=float)
    c = df[f"{metric}_C"].to_numpy(dtype=float)
    d = df[f"{metric}_D"].to_numpy(dtype=float)
    mean_ratio, ratio_samples = bootstrap_ratio(
        b,
        c,
        d,
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "n_instances": int(df.shape[0]),
        "mean_B": safe_mean(b),
        "mean_C": safe_mean(c),
        "mean_D": safe_mean(d),
        "recovery_ratio": None if mean_ratio is None else float(mean_ratio),
        "recovery_ratio_ci95": ci95(ratio_samples),
    }


def summarize_by_gap_stratum(
    merged: pd.DataFrame,
    metric: str,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    out = {}
    for stratum, sub_df in merged.groupby("gap_stratum", dropna=False):
        key = "unknown" if pd.isna(stratum) else str(stratum)
        out[key] = summarize_recovery(
            sub_df,
            metric=metric,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze Stage 1 recovery metrics")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = get_stage1_paths(args.model_id)
    paths.analysis.mkdir(parents=True, exist_ok=True)
    paths.plots.mkdir(parents=True, exist_ok=True)

    run_config = load_json(paths.run_config)
    if not isinstance(run_config, dict):
        run_config = {
            "missing_run_config": True,
            "model_id": args.model_id,
            "seed": args.seed,
        }

    b_df = load_eval_csv(paths.evaluation / "condition_B.per_instance.csv")
    c_df = load_eval_csv(paths.evaluation / "condition_C.per_instance.csv")
    d_df = load_eval_csv(paths.evaluation / "condition_D.per_instance.csv")

    merged = (
        b_df[["instance_id", "pass_at_1_proxy", "bleu4", "gap_stratum", "bleu_gap_bc"]]
        .rename(
            columns={
                "pass_at_1_proxy": "pass_B",
                "bleu4": "bleu_B",
            }
        )
        .merge(
            c_df[["instance_id", "pass_at_1_proxy", "bleu4"]].rename(
                columns={
                    "pass_at_1_proxy": "pass_C",
                    "bleu4": "bleu_C",
                }
            ),
            on="instance_id",
            how="inner",
        )
        .merge(
            d_df[["instance_id", "pass_at_1_proxy", "bleu4"]].rename(
                columns={
                    "pass_at_1_proxy": "pass_D",
                    "bleu4": "bleu_D",
                }
            ),
            on="instance_id",
            how="inner",
        )
    )

    pass_ratio = []
    bleu_ratio = []
    for _, row in merged.iterrows():
        den_pass = row["pass_C"] - row["pass_B"]
        den_bleu = row["bleu_C"] - row["bleu_B"]
        pass_ratio.append(
            None
            if abs(den_pass) < 1e-12
            else (row["pass_D"] - row["pass_B"]) / den_pass
        )
        bleu_ratio.append(
            None
            if abs(den_bleu) < 1e-12
            else (row["bleu_D"] - row["bleu_B"]) / den_bleu
        )

    merged["recovery_ratio_pass"] = pass_ratio
    merged["recovery_ratio_bleu"] = bleu_ratio
    merged["run_model_id"] = run_config.get("model_id", args.model_id)
    merged["run_seed"] = run_config.get("seed", args.seed)
    merged["run_truncation_budget_tokens"] = run_config.get(
        "truncation_budget_tokens", np.nan
    )

    pass_summary = summarize_recovery(
        merged,
        metric="pass",
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        seed=args.seed,
    )
    bleu_summary = summarize_recovery(
        merged,
        metric="bleu",
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        seed=args.seed,
    )

    summary = {
        "n_instances": int(merged.shape[0]),
        "mean_pass_B": pass_summary["mean_B"],
        "mean_pass_C": pass_summary["mean_C"],
        "mean_pass_D": pass_summary["mean_D"],
        "mean_bleu_B": bleu_summary["mean_B"],
        "mean_bleu_C": bleu_summary["mean_C"],
        "mean_bleu_D": bleu_summary["mean_D"],
        "recovery_ratio_pass": pass_summary["recovery_ratio"],
        "recovery_ratio_pass_ci95": pass_summary["recovery_ratio_ci95"],
        "recovery_ratio_bleu": bleu_summary["recovery_ratio"],
        "recovery_ratio_bleu_ci95": bleu_summary["recovery_ratio_ci95"],
        "recovery_by_gap_stratum": {
            "pass": summarize_by_gap_stratum(
                merged,
                metric="pass",
                bootstrap_samples=BOOTSTRAP_SAMPLES,
                seed=args.seed,
            ),
            "bleu": summarize_by_gap_stratum(
                merged,
                metric="bleu",
                bootstrap_samples=BOOTSTRAP_SAMPLES,
                seed=args.seed,
            ),
        },
        "run_config": run_config,
    }

    merged.to_csv(paths.analysis / "recovery_per_instance.csv", index=False)
    with (paths.analysis / "recovery_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    metadata = {
        "run_config": run_config,
        "analysis_summary": {
            "n_instances": summary["n_instances"],
            "recovery_ratio_bleu": summary["recovery_ratio_bleu"],
            "recovery_ratio_pass": summary["recovery_ratio_pass"],
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(
        ["B", "C", "D"],
        [summary["mean_pass_B"], summary["mean_pass_C"], summary["mean_pass_D"]],
    )
    axes[0].set_title("Pass@1 Proxy Mean")
    axes[0].set_ylim(0, 1)

    axes[1].bar(
        ["B", "C", "D"],
        [summary["mean_bleu_B"], summary["mean_bleu_C"], summary["mean_bleu_D"]],
    )
    axes[1].set_title("BLEU-4 Mean")
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    means_plot = paths.plots / "condition_means.png"
    fig.savefig(
        means_plot,
        dpi=160,
        metadata={"run_config": json.dumps(run_config, sort_keys=True)},
    )
    plt.close(fig)
    with (paths.plots / "condition_means.meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    bleu_vals = [x for x in bleu_ratio if x is not None and np.isfinite(x)]
    if bleu_vals:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(bleu_vals, bins=20)
        ax2.set_title("Recovery Ratio Distribution (BLEU)")
        ax2.set_xlabel("(D-B)/(C-B)")
        ax2.set_ylabel("Count")
        fig2.tight_layout()
        hist_plot = paths.plots / "recovery_ratio_bleu_hist.png"
        fig2.savefig(
            hist_plot,
            dpi=160,
            metadata={"run_config": json.dumps(run_config, sort_keys=True)},
        )
        plt.close(fig2)
        with (paths.plots / "recovery_ratio_bleu_hist.meta.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f, indent=2)

    print("Saved analysis summary and plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
