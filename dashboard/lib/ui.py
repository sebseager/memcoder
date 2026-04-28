from __future__ import annotations

import html
import json
from typing import Any

import streamlit as st

from dashboard.lib import runtime
from dashboard.lib.data import list_difficulties, list_repo_ids, load_documents, load_repo_meta


def page_setup(title: str) -> tuple[str, str]:
    st.set_page_config(page_title=f"SHINE Dashboard - {title}", layout="wide")
    return render_sidebar()


def render_sidebar() -> tuple[str, str]:
    runtime.update_lora_load_status()

    repos = list_repo_ids()
    if not repos:
        st.sidebar.error("No artifact repos found under artifacts/.")
        st.stop()

    current_repo = st.session_state.get("selected_repo")
    repo_index = repos.index(current_repo) if current_repo in repos else 0
    selected_repo = st.sidebar.selectbox("Repository", repos, index=repo_index, key="selected_repo")
    runtime.sync_repo_selection(selected_repo)

    difficulties = list_difficulties(selected_repo)
    current_difficulty = st.session_state.get("selected_difficulty", "all")
    diff_index = difficulties.index(current_difficulty) if current_difficulty in difficulties else 0
    selected_difficulty = st.sidebar.selectbox(
        "Difficulty",
        difficulties,
        index=diff_index,
        key="selected_difficulty",
    )

    docs = load_documents(selected_repo)
    repo_meta = load_repo_meta(selected_repo)
    st.sidebar.caption(f"{len(docs)} documents")
    if repo_meta.get("commit"):
        st.sidebar.caption(f"Commit `{str(repo_meta['commit'])[:12]}`")

    _render_lora_status()
    return selected_repo, selected_difficulty


def _render_lora_status() -> None:
    st.sidebar.divider()
    st.sidebar.markdown("**LoRA status**")
    future = st.session_state.get("lora_future")
    pending_lora_id = st.session_state.get("pending_lora_id")
    loaded_lora_id = st.session_state.get("loaded_lora_id")
    error = st.session_state.get("lora_load_error")

    if future is not None and not future.done():
        st.sidebar.info(f"Loading `{pending_lora_id}`...")
        return
    if error:
        st.sidebar.error(error)
        return
    if loaded_lora_id:
        st.sidebar.success(f"Loaded `{loaded_lora_id}`")
    else:
        st.sidebar.caption("No LoRA loaded")


def lora_label(doc: Any) -> str:
    topic = doc.topic or "Untitled topic"
    suffix = "ready" if doc.lora_exists else "missing file"
    return f"{topic} / {doc.document_id} ({suffix})"


def score_block(judge: dict[str, Any] | None) -> None:
    if not judge:
        st.caption("No judge score available.")
        return
    score = max(0, min(5, int(judge.get("score") or 0)))
    reasoning = str(judge.get("reasoning") or "")
    one_line = reasoning.split(". ")[0].strip()

    score_col, text_col = st.columns([0.9, 2.1], vertical_alignment="center")
    colors = {
        1: "#dc2626",
        2: "#f97316",
        3: "#eab308",
        4: "#2563eb",
        5: "#9333ea",
    }
    segments = []
    for idx in range(1, 6):
        selected = idx == score
        segments.append(
            "<div style='"
            "flex:1;"
            "padding:0.32rem 0;"
            "text-align:center;"
            "font-weight:700;"
            f"background:{colors[idx]};"
            "color:white;"
            f"opacity:{'1' if selected else '0.35'};"
            f"box-shadow:{'inset 0 0 0 2px rgba(255,255,255,0.9)' if selected else 'none'};"
            "'>"
            f"{idx}"
            "</div>"
        )
    with score_col:
        st.markdown(
            "<div style='"
            "display:flex;"
            "overflow:hidden;"
            "border:1px solid rgba(49,51,63,0.25);"
            "border-radius:0.6rem;"
            "background:rgba(49,51,63,0.04);"
            "'>"
            + "".join(segments)
            + "</div>",
            unsafe_allow_html=True,
        )
    with text_col:
        if one_line:
            st.markdown(
                f"<div style='font-size:0.9rem;color:rgba(49,51,63,0.75);'>{html.escape(one_line)}</div>",
                unsafe_allow_html=True,
            )
    with st.expander("Full judge explanation", expanded=False):
        st.write(reasoning or "No reasoning returned.")
        modes = judge.get("failure_modes") or []
        if modes:
            st.write("Failure modes:", ", ".join(modes))


def answer_panel(title: str, answer: dict[str, str] | None, judge: dict[str, Any] | None) -> None:
    with st.container(border=True):
        st.subheader(title)
        if not answer:
            st.caption("No answer generated yet.")
            return
        st.write(answer.get("answer") or answer.get("raw_generation") or "")
        if answer.get("think"):
            with st.expander("Thinking trace", expanded=False):
                st.write(answer["think"])
        score_block(judge)


def json_expander(label: str, payload: Any) -> None:
    with st.expander(label, expanded=False):
        st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")
