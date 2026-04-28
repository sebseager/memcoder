from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.lib import runtime
from dashboard.lib.data import document_by_id, lora_options, question_options
from dashboard.lib.ui import answer_panel, lora_label, page_setup, progress_button


def clear_routing_outputs() -> None:
    st.session_state["routing_result"] = None
    st.session_state["routing_answer"] = None
    st.session_state["routing_judge"] = None


def _similarity_cell_style(value: Any, vmin: float, vmax: float) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "font-weight:700;"
    span = max(vmax - vmin, 1e-9)
    norm = max(0.0, min(1.0, (numeric - vmin) / span))
    low = (219, 234, 254)
    high = (30, 64, 175)
    red = round(low[0] + (high[0] - low[0]) * norm)
    green = round(low[1] + (high[1] - low[1]) * norm)
    blue = round(low[2] + (high[2] - low[2]) * norm)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    text_color = "#111827" if luminance > 150 else "#ffffff"
    return f"background-color: rgb({red}, {green}, {blue}); color: {text_color}; font-weight: 700;"


def _ranked_loras_table(route: dict[str, Any] | None, repo_id: str) -> pd.DataFrame:
    rows = []
    if not route:
        return pd.DataFrame(rows)
    for candidate in route.get("ranked_loras") or []:
        doc = document_by_id(repo_id, str(candidate.get("lora_id")))
        rows.append(
            {
                "rank": candidate.get("rank"),
                "lora_id": candidate.get("lora_id"),
                "title": candidate.get("topic") or (doc.topic if doc else ""),
                "similarity": candidate.get("score"),
            }
        )
    return pd.DataFrame(rows)


def _styled_ranked_loras(df: pd.DataFrame) -> Any:
    if df.empty or "similarity" not in df:
        return df
    scores = pd.to_numeric(df["similarity"], errors="coerce")
    vmin = float(scores.min()) if scores.notna().any() else 0.0
    vmax = float(scores.max()) if scores.notna().any() else 1.0
    return df.style.format({"similarity": "{:.3f}"}).map(
        lambda value: _similarity_cell_style(value, vmin, vmax),
        subset=["similarity"],
    )


def render_outputs(
    answer_slot: Any,
    ranked_slot: Any,
    *,
    repo_id: str,
    judge_pending: bool = False,
) -> None:
    with answer_slot.container():
        answer_panel(
            "Routed SHINE",
            st.session_state.get("routing_answer"),
            st.session_state.get("routing_judge"),
            panel_key="routed-shine",
            judge_pending=judge_pending,
        )

    route = st.session_state.get("routing_result")
    with ranked_slot.container(), st.container(border=True):
        st.subheader("Ranked LoRAs")
        if not route:
            st.caption("No routing result yet.")
        else:
            df = _ranked_loras_table(route, repo_id)
            st.dataframe(
                _styled_ranked_loras(df),
                hide_index=True,
                width="stretch",
                column_config={
                    "rank": st.column_config.NumberColumn("Rank", width="small"),
                    "similarity": st.column_config.NumberColumn("Similarity", width="small"),
                },
            )


repo_id, difficulty = page_setup("Routing Explorer")
docs = lora_options(repo_id, difficulty)

st.title("Routing Explorer + SHINE Answer")
st.caption("Route a question to candidate LoRAs, then answer with the top-ranked LoRA.")

if not docs:
    st.warning("No document entries are available for this repo and difficulty.")
    st.stop()

left, right = st.columns([0.9, 1.6], gap="large")

with left:
    st.subheader("Question")
    source_idx = st.selectbox(
        "Example source",
        range(len(docs)),
        format_func=lambda idx: lora_label(docs[idx]),
    )
    source_doc = docs[source_idx]
    options = question_options(source_doc)
    sample_idx = st.selectbox(
        "Question examples",
        range(len(options) + 1),
        format_func=lambda idx: "Custom question" if idx == 0 else options[idx - 1]["label"],
    )
    selected_option = options[sample_idx - 1] if sample_idx else None
    default_question = selected_option["question"] if selected_option else ""
    expected_answer = selected_option.get("answer") if selected_option else None

    with st.form("routing_question_form", clear_on_submit=False):
        question = st.text_area(
            "Question",
            value=default_question,
            height=120,
            key=f"routing_question_{source_doc.document_id}_{sample_idx}",
        )
        top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
        embedding_model = st.text_input("Embedding model", value=runtime.DEFAULT_EMBEDDING_MODEL)
        embedding_device = st.selectbox(
            "Embedding device",
            ["cpu", "cuda"],
            index=0 if runtime.DEFAULT_EMBEDDING_DEVICE == "cpu" else 1,
            help="CPU keeps the router from competing with the SHINE model for GPU memory.",
        )
        button_slot = st.empty()
        with button_slot:
            submit = st.form_submit_button(
                "Route and Answer",
                type="primary",
                width="stretch",
                on_click=clear_routing_outputs,
            )

with right:
    answer_slot = st.empty()
    ranked_slot = st.empty()
    render_outputs(answer_slot, ranked_slot, repo_id=repo_id)

if submit:
    st.session_state["routing_result"] = None
    st.session_state["routing_answer"] = None
    st.session_state["routing_judge"] = None
    render_outputs(answer_slot, ranked_slot, repo_id=repo_id)
    try:
        progress_button(button_slot, "Routing with the embedding router...")
        runtime.set_embedding_model_status(
            "loading",
            model_name=embedding_model,
            device=embedding_device,
        )
        route = runtime.route_question_cached(
            repo_id,
            question,
            top_k,
            embedding_model,
            embedding_device,
        )
        runtime.set_embedding_model_status(
            "loaded",
            model_name=str(route.get("embedding_model") or embedding_model),
            device=str(route.get("embedding_device") or embedding_device),
        )
        st.session_state["routing_result"] = route
        render_outputs(answer_slot, ranked_slot, repo_id=repo_id)

        top = (route.get("ranked_loras") or [])[0]
        routed_doc = document_by_id(repo_id, str(top.get("lora_id")))
        if routed_doc is None:
            raise RuntimeError(f"Router returned unknown LoRA id: {top.get('lora_id')}")
        progress_button(button_slot, f"Generating SHINE answer with `{routed_doc.document_id}`...")
        answer = runtime.generate_answer_cached(
            repo_id=repo_id,
            condition="shine",
            question=question,
            document_text=routed_doc.doc_text,
            lora_id=routed_doc.document_id,
            lora_path=routed_doc.lora_path,
        )
        st.session_state["routing_answer"] = answer
        render_outputs(
            answer_slot,
            ranked_slot,
            repo_id=repo_id,
            judge_pending=bool(expected_answer),
        )
        if expected_answer:
            progress_button(button_slot, "Judging SHINE answer...")
            st.session_state["routing_judge"] = runtime.judge_answer_cached(
                repo_id,
                question,
                answer.get("answer", ""),
                str(expected_answer),
                source_doc.document_id,
                source_doc.doc_text,
            )
            render_outputs(answer_slot, ranked_slot, repo_id=repo_id)
    except Exception as exc:  # noqa: BLE001
        if st.session_state.get("routing_result") is None:
            runtime.set_embedding_model_status(
                "error",
                model_name=embedding_model,
                device=embedding_device,
                error=str(exc),
            )
        st.session_state["routing_answer"] = {
            "answer": f"Unable to complete routed SHINE answer: {exc}",
            "raw_generation": "",
            "think": "",
        }
        render_outputs(answer_slot, ranked_slot, repo_id=repo_id)
    st.rerun()
