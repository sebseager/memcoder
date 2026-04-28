from __future__ import annotations

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
    score = int(judge.get("score") or 0)
    cols = st.columns(5)
    for idx, col in enumerate(cols, start=1):
        if idx == score:
            col.markdown(
                f"<div style='padding:0.35rem;text-align:center;border-radius:0.35rem;"
                f"background:#1f77b4;color:white;font-weight:700'>{idx}</div>",
                unsafe_allow_html=True,
            )
        else:
            col.markdown(
                f"<div style='padding:0.35rem;text-align:center;border-radius:0.35rem;"
                f"border:1px solid #ddd;color:#666'>{idx}</div>",
                unsafe_allow_html=True,
            )
    reasoning = str(judge.get("reasoning") or "")
    one_line = reasoning.split(". ")[0].strip()
    if one_line:
        st.caption(one_line)
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
