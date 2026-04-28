from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.lib import runtime
from dashboard.lib.data import document_by_id, lora_options, question_options
from dashboard.lib.ui import answer_panel, lora_label, page_setup


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
    question = st.text_area(
        "Question",
        value=default_question,
        height=120,
        key=f"routing_question_{source_doc.document_id}_{sample_idx}",
    )
    expected_answer = selected_option.get("answer") if selected_option else None
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
    embedding_model = st.text_input("Embedding model", value=runtime.DEFAULT_EMBEDDING_MODEL)
    embedding_device = st.selectbox(
        "Embedding device",
        ["cpu", "cuda"],
        index=0 if runtime.DEFAULT_EMBEDDING_DEVICE == "cpu" else 1,
        help="CPU keeps the router from competing with the SHINE model for GPU memory.",
    )
    submit = st.button("Route and Answer", type="primary", width="stretch")

if submit:
    st.session_state["routing_result"] = None
    st.session_state["routing_answer"] = None
    st.session_state["routing_judge"] = None
    try:
        with st.spinner("Routing with the embedding router..."):
            route = runtime.route_question_cached(
                repo_id,
                question,
                top_k,
                embedding_model,
                embedding_device,
            )
        st.session_state["routing_result"] = route

        top = (route.get("ranked_loras") or [])[0]
        routed_doc = document_by_id(repo_id, str(top.get("lora_id")))
        if routed_doc is None:
            raise RuntimeError(f"Router returned unknown LoRA id: {top.get('lora_id')}")
        with st.spinner(f"Generating SHINE answer with `{routed_doc.document_id}`..."):
            answer = runtime.generate_answer_cached(
                repo_id=repo_id,
                condition="shine",
                question=question,
                document_text=routed_doc.doc_text,
                lora_id=routed_doc.document_id,
                lora_path=routed_doc.lora_path,
            )
        st.session_state["routing_answer"] = answer
        if expected_answer:
            with st.spinner("Judging SHINE answer..."):
                st.session_state["routing_judge"] = runtime.judge_answer_cached(
                    repo_id,
                    question,
                    answer.get("answer", ""),
                    str(expected_answer),
                )
    except Exception as exc:  # noqa: BLE001
        st.session_state["routing_answer"] = {
            "answer": f"Unable to complete routed SHINE answer: {exc}",
            "raw_generation": "",
            "think": "",
        }

with right:
    route = st.session_state.get("routing_result")
    answer = st.session_state.get("routing_answer")
    judge = st.session_state.get("routing_judge")

    answer_panel("Routed SHINE", answer, judge)

    st.subheader("Ranked LoRAs")
    if not route:
        st.caption("No routing result yet.")
    else:
        rows = []
        for candidate in route.get("ranked_loras") or []:
            doc = document_by_id(repo_id, str(candidate.get("lora_id")))
            rows.append(
                {
                    "rank": candidate.get("rank"),
                    "lora_id": candidate.get("lora_id"),
                    "title": candidate.get("topic") or (doc.topic if doc else ""),
                    "similarity": candidate.get("score"),
                    "description": doc.description if doc else "",
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
