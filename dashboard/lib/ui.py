from __future__ import annotations

import html
import json
from typing import Any

import streamlit as st

from dashboard.lib import runtime
from dashboard.lib.data import list_difficulties, list_repo_ids, load_documents, load_repo_meta


def page_setup(title: str) -> tuple[str, str]:
    st.set_page_config(page_title=f"SHINE Dashboard - {title}", layout="wide")
    _inject_dashboard_css()
    return render_sidebar()


def render_sidebar() -> tuple[str, str]:
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

    with st.sidebar:
        _render_model_statuses()
    return selected_repo, selected_difficulty


@st.fragment(run_every="1s")
def _render_model_statuses() -> None:
    runtime.update_lora_load_status()

    st.divider()
    st.markdown("**LoRA status**")
    future = st.session_state.get("lora_future")
    pending_lora_id = st.session_state.get("pending_lora_id")
    loaded_lora_id = st.session_state.get("loaded_lora_id")
    error = st.session_state.get("lora_load_error")

    if future is not None and not future.done():
        st.info(f"Loading `{pending_lora_id}`...")
    elif error:
        st.error(error)
    elif loaded_lora_id:
        st.success(f"Loaded `{loaded_lora_id}`")
    else:
        st.caption("No LoRA loaded")

    st.markdown("**Embedding model**")
    embedding_status = st.session_state.get("embedding_model_status") or {}
    state = embedding_status.get("state")
    model_name = embedding_status.get("model_name")
    device = embedding_status.get("device")
    if state == "loading" and model_name:
        st.info(f"Loading `{model_name}` on `{device or 'auto'}`...")
    elif state == "loaded" and model_name:
        suffix = f" on `{device}`" if device else ""
        st.success(f"Loaded `{model_name}`{suffix}")
    elif state == "error":
        st.error(str(embedding_status.get("error") or "Embedding model failed to load."))
    else:
        st.caption("No embedding model loaded")


def lora_label(doc: Any) -> str:
    topic = doc.topic or "Untitled topic"
    suffix = "ready" if doc.lora_exists else "missing file"
    return f"{topic} / {doc.document_id} ({suffix})"


def _inject_dashboard_css() -> None:
    st.markdown(
        """
        <style>
        .mc-progress-button {
            align-items: center;
            background: #ff4b4b;
            border: 1px solid #ff4b4b;
            border-radius: 0.5rem;
            color: white;
            cursor: not-allowed;
            display: inline-flex;
            font-weight: 600;
            gap: 0.5rem;
            justify-content: center;
            line-height: 1.6;
            min-height: 2.5rem;
            opacity: 0.72;
            padding: 0.375rem 0.75rem;
            width: 100%;
        }
        .mc-button-spinner {
            animation: mc-spin 0.8s linear infinite;
            border: 2px solid rgba(255, 255, 255, 0.45);
            border-radius: 50%;
            border-top-color: white;
            height: 0.9rem;
            width: 0.9rem;
        }
        @keyframes mc-spin {
            to { transform: rotate(360deg); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def progress_button(slot: Any, label: str) -> None:
    slot.markdown(
        "<button class='mc-progress-button' disabled>"
        "<span class='mc-button-spinner' aria-hidden='true'></span>"
        f"<span>{html.escape(label)}</span>"
        "</button>",
        unsafe_allow_html=True,
    )


def score_block(judge: dict[str, Any] | None) -> None:
    if not judge:
        st.caption("No judge score available.")
        return
    score = max(0, min(5, int(judge.get("score") or 0)))
    reasoning = str(judge.get("reasoning") or "")
    one_line = reasoning.split(". ")[0].strip()

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
        radius = ""
        if idx == 1:
            radius = "border-radius:0.48rem 0 0 0.48rem;"
        elif idx == 5:
            radius = "border-radius:0 0.48rem 0.48rem 0;"
        segments.append(
            "<div style='"
            "flex:1;"
            "padding:0.26rem 0;"
            "text-align:center;"
            "font-weight:700;"
            f"background:{colors[idx]};"
            "color:white;"
            f"opacity:{'1' if selected else '0.35'};"
            f"{radius}"
            f"box-shadow:{'inset 0 0 0 2px rgba(255,255,255,0.95)' if selected else 'none'};"
            "'>"
            f"{idx}"
            "</div>"
        )
    summary = (
        f"<div style='font-size:0.9rem;color:white;'>{html.escape(one_line)}</div>"
        if one_line
        else ""
    )
    st.markdown(
        "<div style='display:flex;align-items:center;gap:0.55rem;'>"
        "<div style='"
        "display:flex;"
        "flex:0 0 13.5rem;"
        "overflow:hidden;"
        "border:1px solid rgba(49,51,63,0.25);"
        "border-radius:0.6rem;"
        "background:rgba(49,51,63,0.04);"
        "'>"
        + "".join(segments)
        + "</div>"
        + summary
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
    with st.expander("Full judge explanation", expanded=False):
        st.write(reasoning or "No reasoning returned.")
        modes = judge.get("failure_modes") or []
        if modes:
            st.write("Failure modes:", ", ".join(modes))


def answer_panel(
    title: str,
    answer: dict[str, str] | None,
    judge: dict[str, Any] | None,
    *,
    judge_pending: bool = False,
) -> None:
    with st.container(border=True):
        st.subheader(title)
        if not answer:
            st.caption("No answer generated yet.")
            return
        st.write(answer.get("answer") or answer.get("raw_generation") or "")
        if answer.get("think"):
            with st.expander("Thinking trace", expanded=False):
                st.write(answer["think"])
        if judge_pending:
            st.caption("Judging answer...")
            return
        score_block(judge)


def json_expander(label: str, payload: Any) -> None:
    with st.expander(label, expanded=False):
        st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")
