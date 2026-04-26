"""
Exp 0 — Dataset Characterization for DyPRAG

For every instance in SWE-Bench Lite (test split), tokenize the ground-truth
relevant files and sum their token lengths using the Qwen3-8B tokenizer.

Produces:
  1. A histogram of total relevant-file token counts across instances
  2. The context-constrained subset (sum > budget tokens)
  3. The unconstrained subset (sum <= budget tokens)
  4. Automatic budget threshold adjustment if constrained subset < 80 or > 120

Outputs are written to exp_0/results/.
"""

import json
import re
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-8B"
DEFAULT_BUDGET = 4096
TARGET_CONSTRAINED_MIN = 80
TARGET_CONSTRAINED_MAX = 120
RESULTS_DIR = Path(__file__).parent / "results"
CACHE_DIR = Path(__file__).parent / ".file_cache"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_patched_files(patch: str) -> list[str]:
    """Extract file paths from a unified diff patch."""
    return re.findall(r"diff --git a/(.*?) b/", patch)


def fetch_file_content(repo: str, commit: str, filepath: str) -> str | None:
    """Fetch a file from GitHub at a specific commit, with local caching."""
    cache_key = f"{repo}/{commit}/{filepath}".replace("/", "__")
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace")

    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{filepath}"
    req = Request(url, headers={"User-Agent": "dyprag-exp0"})
    try:
        with urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8", errors="replace")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(content, encoding="utf-8")
        return content
    except HTTPError as e:
        print(f"  WARN: HTTP {e.code} fetching {url}")
        return None
    except Exception as e:
        print(f"  WARN: error fetching {url}: {e}")
        return None


def tokenize_and_count(tokenizer, text: str) -> int:
    """Return token count for a text string."""
    return len(tokenizer.encode(text, add_special_tokens=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading SWE-bench Lite test split...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    print(f"  {len(ds)} instances, {len(set(ds['repo']))} unique repos")

    print(f"Loading tokenizer: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # --- Collect token counts per instance ---
    records = []
    for i, row in enumerate(ds):
        instance_id = row["instance_id"]
        repo = row["repo"]
        commit = row["base_commit"]
        files = extract_patched_files(row["patch"])

        total_tokens = 0
        file_details = []
        for fpath in files:
            content = fetch_file_content(repo, commit, fpath)
            if content is None:
                file_details.append({"path": fpath, "tokens": None, "error": True})
                continue
            n_tokens = tokenize_and_count(tokenizer, content)
            total_tokens += n_tokens
            file_details.append({"path": fpath, "tokens": n_tokens})

        records.append(
            {
                "instance_id": instance_id,
                "repo": repo,
                "base_commit": commit,
                "files": file_details,
                "total_tokens": total_tokens,
                "n_files": len(files),
            }
        )

        if (i + 1) % 25 == 0 or i == 0:
            print(
                f"  [{i + 1}/{len(ds)}] {instance_id}: {total_tokens} tokens ({len(files)} files)"
            )

    # Save raw data
    raw_path = RESULTS_DIR / "token_counts.json"
    with open(raw_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nRaw data saved to {raw_path}")

    # --- Analyze at multiple budget thresholds ---
    token_counts = np.array([r["total_tokens"] for r in records])

    # Coarse sweep first, then fine sweep around candidates
    coarse_thresholds = [2048, 3072, 4096, 6144, 8192, 10000, 12000, 14000, 16384]
    fine_thresholds = list(range(8000, 16001, 500))
    all_thresholds = sorted(set(coarse_thresholds + fine_thresholds))

    print("\n=== Budget Threshold Sweep ===")
    print(
        f"{'Budget':>8} | {'Constrained':>12} | {'Unconstrained':>14} | {'Status':>10}"
    )
    print("-" * 55)

    best_budget = DEFAULT_BUDGET
    best_count = None
    target_mid = (TARGET_CONSTRAINED_MIN + TARGET_CONSTRAINED_MAX) / 2

    for budget in all_thresholds:
        n_constrained = int(np.sum(token_counts > budget))
        n_unconstrained = int(np.sum(token_counts <= budget))
        in_range = TARGET_CONSTRAINED_MIN <= n_constrained <= TARGET_CONSTRAINED_MAX
        status = "  OK" if in_range else ""
        print(
            f"{budget:>8} | {n_constrained:>12} | {n_unconstrained:>14} | {status:>10}"
        )
        if in_range:
            if best_count is None or abs(n_constrained - target_mid) < abs(
                best_count - target_mid
            ):
                best_budget = budget
                best_count = n_constrained

    # If no threshold hits the target range, pick the one closest
    if best_count is None:
        diffs = []
        for budget in all_thresholds:
            n = int(np.sum(token_counts > budget))
            diffs.append((abs(n - target_mid), budget, n))
        diffs.sort()
        best_budget = diffs[0][1]
        best_count = diffs[0][2]
        print(
            f"\nNo threshold in [{TARGET_CONSTRAINED_MIN}, {TARGET_CONSTRAINED_MAX}] range."
        )
        print(f"Closest: budget={best_budget} -> {best_count} constrained instances.")
    else:
        print(
            f"\nSelected budget: {best_budget} tokens -> {best_count} constrained instances."
        )

    # --- Build subsets ---
    constrained = [r for r in records if r["total_tokens"] > best_budget]
    unconstrained = [r for r in records if r["total_tokens"] <= best_budget]

    subsets = {
        "budget_threshold": best_budget,
        "constrained_count": len(constrained),
        "unconstrained_count": len(unconstrained),
        "constrained_instance_ids": [r["instance_id"] for r in constrained],
        "unconstrained_instance_ids": [r["instance_id"] for r in unconstrained],
    }
    subsets_path = RESULTS_DIR / "subsets.json"
    with open(subsets_path, "w") as f:
        json.dump(subsets, f, indent=2)
    print(f"Subsets saved to {subsets_path}")

    # --- Summary statistics ---
    summary = {
        "total_instances": len(records),
        "budget_threshold": best_budget,
        "constrained_count": len(constrained),
        "unconstrained_count": len(unconstrained),
        "token_count_stats": {
            "min": int(np.min(token_counts)),
            "max": int(np.max(token_counts)),
            "mean": float(np.mean(token_counts)),
            "median": float(np.median(token_counts)),
            "p25": float(np.percentile(token_counts, 25)),
            "p75": float(np.percentile(token_counts, 75)),
            "p90": float(np.percentile(token_counts, 90)),
            "p95": float(np.percentile(token_counts, 95)),
        },
        "per_repo_stats": {},
    }
    for repo in sorted(set(ds["repo"])):
        repo_tokens = [r["total_tokens"] for r in records if r["repo"] == repo]
        summary["per_repo_stats"][repo] = {
            "count": len(repo_tokens),
            "mean_tokens": float(np.mean(repo_tokens)),
            "median_tokens": float(np.median(repo_tokens)),
            "constrained": int(sum(1 for t in repo_tokens if t > best_budget)),
        }

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # --- Print summary ---
    print("\n=== Summary ===")
    stats = summary["token_count_stats"]
    print(
        f"Token counts: min={stats['min']}, median={stats['median']:.0f}, "
        f"mean={stats['mean']:.0f}, p75={stats['p75']:.0f}, "
        f"p95={stats['p95']:.0f}, max={stats['max']}"
    )
    print(f"Budget threshold: {best_budget} tokens")
    print(f"Constrained (>{best_budget}): {len(constrained)} instances")
    print(f"Unconstrained (≤{best_budget}): {len(unconstrained)} instances")

    print("\nPer-repo breakdown:")
    for repo, rs in sorted(summary["per_repo_stats"].items()):
        print(
            f"  {repo:>30}: {rs['count']:>3} instances, "
            f"median={rs['median_tokens']:>6.0f} tokens, "
            f"{rs['constrained']:>3} constrained"
        )

    # --- Generate histogram ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of token counts
    ax = axes[0]
    ax.hist(token_counts, bins=50, edgecolor="black", alpha=0.7, color="#4C72B0")
    ax.axvline(
        best_budget,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Budget = {best_budget}",
    )
    ax.set_xlabel("Total tokens in relevant file(s)")
    ax.set_ylabel("Number of instances")
    ax.set_title("SWE-Bench Lite: Token Count Distribution")
    ax.legend()

    # CDF
    ax = axes[1]
    sorted_counts = np.sort(token_counts)
    cdf = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    ax.plot(sorted_counts, cdf, color="#4C72B0", linewidth=2)
    ax.axvline(
        best_budget,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Budget = {best_budget}",
    )
    ax.axhline(
        len(unconstrained) / len(records),
        color="gray",
        linestyle=":",
        linewidth=1,
        alpha=0.7,
    )
    ax.set_xlabel("Total tokens in relevant file(s)")
    ax.set_ylabel("Cumulative fraction of instances")
    ax.set_title("CDF of Token Counts")
    ax.legend()

    plt.tight_layout()
    hist_path = RESULTS_DIR / "token_distribution.png"
    fig.savefig(hist_path, dpi=150)
    plt.close()
    print(f"\nHistogram saved to {hist_path}")

    # --- Histogram at key thresholds ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(token_counts, bins=50, edgecolor="black", alpha=0.7, color="#4C72B0")
    key_thresholds = [4096, 8192, best_budget, 14000]
    key_thresholds = sorted(set(key_thresholds))
    colors = ["#E74C3C", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
    for budget, color in zip(key_thresholds, colors):
        n_c = int(np.sum(token_counts > budget))
        style = "-" if budget == best_budget else "--"
        lw = 2.5 if budget == best_budget else 1.5
        ax.axvline(
            budget,
            color=color,
            linestyle=style,
            linewidth=lw,
            label=f"{budget} tok ({n_c} constrained)"
            + (" [selected]" if budget == best_budget else ""),
        )
    ax.set_xlabel("Total tokens in relevant file(s)")
    ax.set_ylabel("Number of instances")
    ax.set_title("SWE-Bench Lite: Token Counts with Budget Thresholds")
    ax.legend(fontsize=8)
    plt.tight_layout()
    sweep_path = RESULTS_DIR / "threshold_sweep.png"
    fig.savefig(sweep_path, dpi=150)
    plt.close()
    print(f"Threshold sweep chart saved to {sweep_path}")

    # --- Per-repo box plot ---
    fig, ax = plt.subplots(figsize=(12, 5))
    repo_names = sorted(set(ds["repo"]))
    repo_data = [
        [r["total_tokens"] for r in records if r["repo"] == name] for name in repo_names
    ]
    short_names = [r.split("/")[1] for r in repo_names]
    bp = ax.boxplot(repo_data, tick_labels=short_names, vert=True, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.6)
    ax.axhline(
        best_budget,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Budget = {best_budget}",
    )
    ax.set_ylabel("Total tokens in relevant file(s)")
    ax.set_title("Token Counts by Repository")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    repo_path = RESULTS_DIR / "per_repo_boxplot.png"
    fig.savefig(repo_path, dpi=150)
    plt.close()
    print(f"Per-repo boxplot saved to {repo_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
