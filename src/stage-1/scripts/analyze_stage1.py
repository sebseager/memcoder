from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import ANALYSIS_DIR, BOOTSTRAP_SAMPLES, EVAL_DIR, PLOTS_DIR


def bootstrap_ratio(
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    samples: int,
    seed: int = 42,
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
    cols = ["instance_id", "pass_at_1_proxy", "bleu4"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=cols)

    for col in cols:
        if col not in df.columns:
            df[col] = []
    return df[cols]


def safe_mean(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def main() -> int:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    b_df = load_eval_csv(EVAL_DIR / "condition_B.per_instance.csv")
    c_df = load_eval_csv(EVAL_DIR / "condition_C.per_instance.csv")
    d_df = load_eval_csv(EVAL_DIR / "condition_D.per_instance.csv")

    merged = (
        b_df[["instance_id", "pass_at_1_proxy", "bleu4"]]
        .rename(columns={"pass_at_1_proxy": "pass_B", "bleu4": "bleu_B"})
        .merge(
            c_df[["instance_id", "pass_at_1_proxy", "bleu4"]].rename(
                columns={"pass_at_1_proxy": "pass_C", "bleu4": "bleu_C"}
            ),
            on="instance_id",
            how="inner",
        )
        .merge(
            d_df[["instance_id", "pass_at_1_proxy", "bleu4"]].rename(
                columns={"pass_at_1_proxy": "pass_D", "bleu4": "bleu_D"}
            ),
            on="instance_id",
            how="inner",
        )
    )

    pass_ratio = []
    bleu_ratio = []
    for _, r in merged.iterrows():
        den_pass = r["pass_C"] - r["pass_B"]
        den_bleu = r["bleu_C"] - r["bleu_B"]
        pass_ratio.append(
            None if abs(den_pass) < 1e-12 else (r["pass_D"] - r["pass_B"]) / den_pass
        )
        bleu_ratio.append(
            None if abs(den_bleu) < 1e-12 else (r["bleu_D"] - r["bleu_B"]) / den_bleu
        )

    merged["recovery_ratio_pass"] = pass_ratio
    merged["recovery_ratio_bleu"] = bleu_ratio

    b = merged["pass_B"].to_numpy(dtype=float)
    c = merged["pass_C"].to_numpy(dtype=float)
    d = merged["pass_D"].to_numpy(dtype=float)
    pass_boot_mean, pass_samples = bootstrap_ratio(b, c, d, BOOTSTRAP_SAMPLES)

    b2 = merged["bleu_B"].to_numpy(dtype=float)
    c2 = merged["bleu_C"].to_numpy(dtype=float)
    d2 = merged["bleu_D"].to_numpy(dtype=float)
    bleu_boot_mean, bleu_samples = bootstrap_ratio(b2, c2, d2, BOOTSTRAP_SAMPLES)

    def ci(vals: list[float]) -> list[float] | None:
        if not vals:
            return None
        return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]

    summary = {
        "n_instances": int(merged.shape[0]),
        "mean_pass_B": safe_mean(b),
        "mean_pass_C": safe_mean(c),
        "mean_pass_D": safe_mean(d),
        "mean_bleu_B": safe_mean(b2),
        "mean_bleu_C": safe_mean(c2),
        "mean_bleu_D": safe_mean(d2),
        "recovery_ratio_pass": None
        if pass_boot_mean is None
        else float(pass_boot_mean),
        "recovery_ratio_pass_ci95": ci(pass_samples),
        "recovery_ratio_bleu": None
        if bleu_boot_mean is None
        else float(bleu_boot_mean),
        "recovery_ratio_bleu_ci95": ci(bleu_samples),
    }

    merged.to_csv(ANALYSIS_DIR / "recovery_per_instance.csv", index=False)
    with (ANALYSIS_DIR / "recovery_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plot 1: condition means for pass@1 proxy and BLEU.
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
    fig.savefig(PLOTS_DIR / "condition_means.png", dpi=160)
    plt.close(fig)

    # Plot 2: recovery ratio histogram for BLEU.
    bleu_vals = [x for x in bleu_ratio if x is not None and np.isfinite(x)]
    if bleu_vals:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(bleu_vals, bins=20)
        ax2.set_title("Recovery Ratio Distribution (BLEU)")
        ax2.set_xlabel("(D-B)/(C-B)")
        ax2.set_ylabel("Count")
        fig2.tight_layout()
        fig2.savefig(PLOTS_DIR / "recovery_ratio_bleu_hist.png", dpi=160)
        plt.close(fig2)

    print("Saved analysis summary and plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
