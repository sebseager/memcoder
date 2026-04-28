from __future__ import annotations

import streamlit as st

from dashboard.lib import runtime
from dashboard.lib.data import lora_options, question_options, split_sentences
from dashboard.lib.ui import answer_panel, lora_label, page_setup


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
    for sentence in split_sentences(selected_doc.doc_text):
        st.markdown(f"- {sentence}")

    options = question_options(selected_doc)
    sample_idx = st.selectbox(
        "Question examples",
        range(len(options) + 1),
        format_func=lambda idx: "Custom question" if idx == 0 else options[idx - 1]["label"],
    )
    selected_option = options[sample_idx - 1] if sample_idx else None
    default_question = selected_option["question"] if selected_option else ""
    question = st.text_area(
        "Question",
        value=default_question,
        height=120,
        key=f"lora_question_{selected_doc.document_id}_{sample_idx}",
    )
    expected_answer = selected_option.get("answer") if selected_option else None
    if expected_answer:
        with st.expander("Ground truth answer", expanded=False):
            st.write(expected_answer)

    send = st.button("Submit", type="primary", use_container_width=True)

if send:
    st.session_state["side_by_side_answers"] = {}
    st.session_state["side_by_side_judges"] = {}
    for condition, label in (
        ("naive", "Naive"),
        ("in_context", "In-Context"),
        ("shine", "SHINE"),
    ):
        try:
            with st.spinner(f"Generating {label} answer..."):
                answer = runtime.generate_answer_cached(
                    repo_id=repo_id,
                    condition=condition,
                    question=question,
                    document_text=selected_doc.doc_text,
                    lora_id=selected_doc.document_id,
                    lora_path=selected_doc.lora_path,
                )
            st.session_state["side_by_side_answers"][condition] = answer
            if expected_answer:
                with st.spinner(f"Judging {label} answer..."):
                    judge = runtime.judge_answer_cached(
                        repo_id,
                        question,
                        answer.get("answer", ""),
                        str(expected_answer),
                    )
                st.session_state["side_by_side_judges"][condition] = judge
        except Exception as exc:  # noqa: BLE001
            st.session_state["side_by_side_answers"][condition] = {
                "answer": f"Unable to generate {label}: {exc}",
                "raw_generation": "",
                "think": "",
            }

with right:
    st.subheader("Answers")
    answers = st.session_state.get("side_by_side_answers", {})
    judges = st.session_state.get("side_by_side_judges", {})
    answer_panel("Naive", answers.get("naive"), judges.get("naive"))
    answer_panel("In-Context", answers.get("in_context"), judges.get("in_context"))
    answer_panel("SHINE", answers.get("shine"), judges.get("shine"))

    if answers and not expected_answer:
        st.info("Judge scoring is skipped for custom/example questions without a ground-truth answer.")
