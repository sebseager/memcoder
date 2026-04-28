from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def score_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "score" not in df:
        return pd.DataFrame()
    usable = df.dropna(subset=["score"]).copy()
    if usable.empty:
        return pd.DataFrame()
    group_cols = ["condition_display", "routing_strategy", "difficulty", "score_kind"]
    summary = (
        usable.groupby(group_cols, dropna=False)
        .agg(n=("score", "count"), mean_score=("score", "mean"))
        .reset_index()
        .sort_values(group_cols)
    )
    baseline_cols = ["routing_strategy", "difficulty", "score_kind"]
    baselines = (
        summary[summary["condition_display"].isin(["Naive", "Individual LoRA"])]
        .groupby(baseline_cols, dropna=False)["mean_score"]
        .first()
        .rename("baseline_mean")
        .reset_index()
    )
    summary = summary.merge(baselines, on=baseline_cols, how="left")
    summary["delta_vs_baseline"] = summary["mean_score"] - summary["baseline_mean"]
    return summary


def failure_mode_bar(df: pd.DataFrame) -> go.Figure:
    records: list[dict[str, str]] = []
    for _, row in df.iterrows():
        for mode in row.get("failure_modes") or []:
            records.append({"condition": row["condition_display"], "failure_mode": mode})
    if not records:
        return _empty_figure("No failure-mode tags available")
    modes = pd.DataFrame(records)
    counts = modes.groupby(["condition", "failure_mode"]).size().reset_index(name="count")
    return px.bar(
        counts,
        x="condition",
        y="count",
        color="failure_mode",
        barmode="stack",
        title="Failure Modes by Condition",
    )


def topic_heatmap(df: pd.DataFrame) -> go.Figure:
    usable = df.dropna(subset=["score"]).copy()
    if usable.empty:
        return _empty_figure("No scores available")
    pivot = (
        usable.groupby(["topic", "condition_display"], dropna=False)["score"]
        .mean()
        .reset_index()
    )
    if pivot.empty:
        return _empty_figure("No topic scores available")
    table = pivot.pivot(index="topic", columns="condition_display", values="score")
    return px.imshow(
        table,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Mean Score by Topic and Condition",
        labels={"x": "Condition", "y": "Topic", "color": "Mean score"},
    )


def score_distribution(df: pd.DataFrame) -> go.Figure:
    usable = df.dropna(subset=["score"]).copy()
    if usable.empty:
        return _empty_figure("No scores available")
    return px.histogram(
        usable,
        x="score",
        color="condition_display",
        barmode="overlay",
        nbins=10,
        marginal="box",
        title="Score Distribution",
        labels={"condition_display": "Condition"},
    )


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
