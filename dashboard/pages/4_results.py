from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.lib.data import discover_result_logs, load_result_rows
from dashboard.lib.plots import (
    failure_mode_bar,
    score_distribution,
    score_summary,
    topic_heatmap,
)
from dashboard.lib.ui import json_expander, page_setup


repo_id, difficulty = page_setup("Results")

st.title("Results")
st.caption("Aggregate existing JSONL logs for the selected artifact repo. No evaluation is rerun.")

logs = discover_result_logs(repo_id)
if not logs:
    st.warning("No JSONL result logs were found for this repo.")
    st.stop()

default_paths = [log["path"] for log in logs if log["kind"] == "judged"] or [log["path"] for log in logs]
selected_paths = st.multiselect(
    "Result logs",
    options=[log["path"] for log in logs],
    default=default_paths,
    format_func=lambda path: next(log["label"] for log in logs if log["path"] == path),
)
if not selected_paths:
    st.info("Select at least one result log.")
    st.stop()

df = load_result_rows(repo_id, tuple(sorted(selected_paths)))
if difficulty != "all" and "difficulty" in df:
    df = df[df["difficulty"] == difficulty]

if df.empty:
    st.warning("Selected logs have no rows after filtering.")
    st.stop()

st.subheader("Score Summary")
summary = score_summary(df)
if summary.empty:
    st.info("No numeric score or token-F1 values are available.")
else:
    styled_summary = summary.style.format(
        {
            "mean_score": "{:.3f}",
            "baseline_mean": "{:.3f}",
            "delta_vs_baseline": "{:+.3f}",
        }
    )
    if summary["delta_vs_baseline"].notna().any():
        styled_summary = styled_summary.background_gradient(
            subset=["delta_vs_baseline"],
            cmap="RdYlGn",
        )
    st.dataframe(
        styled_summary,
        hide_index=True,
        width="stretch",
    )

chart_cols = st.columns(2)
with chart_cols[0]:
    st.plotly_chart(failure_mode_bar(df), width="stretch")
with chart_cols[1]:
    st.plotly_chart(score_distribution(df), width="stretch")

st.plotly_chart(topic_heatmap(df), width="stretch")

st.subheader("Question-Level Drill-Down")
filter_cols = st.columns(4)
conditions = sorted(df["condition_display"].dropna().unique().tolist())
topics = sorted(df["topic"].dropna().unique().tolist())
selected_conditions = filter_cols[0].multiselect("Condition", conditions, default=conditions)
selected_topics = filter_cols[1].multiselect("Topic", topics, default=topics)
query = filter_cols[2].text_input("Search")
score_kind = filter_cols[3].selectbox("Metric", sorted(df["score_kind"].dropna().unique().tolist()))

filtered = df[
    df["condition_display"].isin(selected_conditions)
    & df["topic"].isin(selected_topics)
    & (df["score_kind"] == score_kind)
].copy()
if query:
    q = query.lower()
    filtered = filtered[
        filtered.apply(lambda row: q in " ".join(str(v).lower() for v in row.values), axis=1)
    ]

display_cols = [
    "source_file",
    "document_id",
    "topic",
    "qa_id",
    "condition_display",
    "routing_strategy",
    "score",
    "question",
]
event = st.dataframe(
    filtered[display_cols],
    hide_index=True,
    width="stretch",
    on_select="rerun",
    selection_mode="single-row",
)

if not filtered.empty:
    selected_row = event.selection.rows[0] if event.selection.rows else 0
    row = filtered.iloc[selected_row]
    with st.expander("Full Log Entry", expanded=True):
        st.markdown("**Question**")
        st.write(row["question"])
        st.markdown("**Ground Truth**")
        st.write(row["expected_answer"])
        st.markdown("**Answer**")
        st.write(row["answer"])
        if row["judge_reasoning"]:
            st.markdown("**Judge**")
            st.write(row["judge_reasoning"])
        json_expander("Routing Trace", row["routing"])
        json_expander("Raw Row", row["raw"])
