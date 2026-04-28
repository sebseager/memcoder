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
from dashboard.lib.data import lora_options, question_options, split_sentences
from dashboard.lib.ui import answer_panel, lora_label, page_setup, progress_button


ANSWER_CONDITIONS = (
    ("naive", "Naive"),
    ("in_context", "In-Context"),
    ("shine", "SHINE"),
)


def render_answer_slots(
    slots: dict[str, Any],
    *,
    note_slot: Any,
    expected_answer: str | None,
    pending_judges: set[str] | None = None,
) -> None:
    answers = st.session_state.get("side_by_side_answers", {})
    judges = st.session_state.get("side_by_side_judges", {})
    pending_judges = pending_judges or set()
    for condition, label in ANSWER_CONDITIONS:
        with slots[condition].container():
            answer_panel(
                label,
                answers.get(condition),
                judges.get(condition),
                judge_pending=condition in pending_judges,
            )

    with note_slot.container():
        if answers and not expected_answer:
            st.info("Judge scoring is skipped for custom/example questions without a ground-truth answer.")


repo_id, difficulty = page_setup("LoRA Explorer")
loras = lora_options(repo_id, difficulty)

st.title("LoRA Explorer + Side-by-Side Answers")
st.caption("Ask one question against the naive, in-context, and SHINE conditions.")

if not loras:
    st.warning("No LoRA/document entries are available for this repo and difficulty.")
    st.stop()

left, right = st.columns([0.9, 1.6], gap="large")

with left:
    st.subheader("Controls")
    selected_idx = st.selectbox(
        "LoRA",
        range(len(loras)),
        format_func=lambda idx: lora_label(loras[idx]),
    )
    selected_doc = loras[selected_idx]
    st.session_state["selected_lora_id"] = selected_doc.document_id

    if selected_doc.lora_path and selected_doc.lora_exists:
        runtime.start_lora_load(repo_id, selected_doc.document_id, selected_doc.lora_path)
    elif selected_doc.lora_path:
        st.warning(f"LoRA path is recorded but missing locally: `{selected_doc.lora_path}`")
    else:
        st.warning("This ledger entry has no `files.lora` path.")

    st.markdown("**Source Document**")
    source_rows = [
        {"row": idx, "text": sentence}
        for idx, sentence in enumerate(split_sentences(selected_doc.doc_text), start=1)
    ]
    if source_rows:
        st.dataframe(
            pd.DataFrame(source_rows),
            hide_index=True,
            width="stretch",
            column_config={
                "row": st.column_config.NumberColumn("Row", width="small"),
            },
        )
    else:
        st.caption("No source document text available.")

with right:
    st.subheader("Question")
    options = question_options(selected_doc)
    sample_idx = st.selectbox(
        "Question examples",
        range(len(options) + 1),
        format_func=lambda idx: "Custom question" if idx == 0 else options[idx - 1]["label"],
    )
    selected_option = options[sample_idx - 1] if sample_idx else None
    default_question = selected_option["question"] if selected_option else ""
    expected_answer = selected_option.get("answer") if selected_option else None
    if expected_answer:
        with st.expander("Ground truth answer", expanded=False):
            st.write(expected_answer)

    with st.form("lora_question_form", clear_on_submit=False):
        question = st.text_area(
            "Question",
            value=default_question,
            height=120,
            key=f"lora_question_{selected_doc.document_id}_{sample_idx}",
        )
        button_slot = st.empty()
        with button_slot:
            send = st.form_submit_button("Submit", type="primary", width="stretch")

    st.subheader("Answers")
    answer_slots = {condition: st.empty() for condition, _ in ANSWER_CONDITIONS}
    judge_note_slot = st.empty()
    render_answer_slots(answer_slots, note_slot=judge_note_slot, expected_answer=expected_answer)

if send:
    st.session_state["side_by_side_answers"] = {}
    st.session_state["side_by_side_judges"] = {}
    render_answer_slots(answer_slots, note_slot=judge_note_slot, expected_answer=expected_answer)
    for condition, label in ANSWER_CONDITIONS:
        try:
            progress_button(button_slot, f"Generating {label} answer...")
            answer = runtime.generate_answer_cached(
                repo_id=repo_id,
                condition=condition,
                question=question,
                document_text=selected_doc.doc_text,
                lora_id=selected_doc.document_id,
                lora_path=selected_doc.lora_path,
            )
            st.session_state["side_by_side_answers"][condition] = answer
            render_answer_slots(
                answer_slots,
                note_slot=judge_note_slot,
                expected_answer=expected_answer,
                pending_judges={condition} if expected_answer else None,
            )
            if expected_answer:
                progress_button(button_slot, f"Judging {label} answer...")
                judge = runtime.judge_answer_cached(
                    repo_id,
                    question,
                    answer.get("answer", ""),
                    str(expected_answer),
                )
                st.session_state["side_by_side_judges"][condition] = judge
                render_answer_slots(answer_slots, note_slot=judge_note_slot, expected_answer=expected_answer)
        except Exception as exc:  # noqa: BLE001
            st.session_state["side_by_side_answers"][condition] = {
                "answer": f"Unable to generate {label}: {exc}",
                "raw_generation": "",
                "think": "",
            }
            render_answer_slots(answer_slots, note_slot=judge_note_slot, expected_answer=expected_answer)
    st.rerun()
