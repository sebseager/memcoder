from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.lib.data import (
    documents_for_difficulty,
    documents_table,
    load_repo_meta,
    qa_table,
)
from dashboard.lib.ui import json_expander, page_setup


repo_id, difficulty = page_setup("Artifact Browser")
docs = documents_for_difficulty(repo_id, difficulty)
meta = load_repo_meta(repo_id)

st.title("Artifact Browser")
st.caption("Generated design documents and QA pairs for the selected artifact repo.")

header_cols = st.columns(3)
header_cols[0].metric("Repo", meta.get("repo_id", repo_id))
header_cols[1].metric("Documents", len(docs))
header_cols[2].metric("Commit", str(meta.get("commit", ""))[:12] or "unknown")

if not docs:
    st.warning("No documents match the selected repo and difficulty.")
    st.stop()

left, right = st.columns([1.25, 1], gap="large")
with left:
    st.subheader("Documents")
    doc_df = documents_table(docs)
    event = st.dataframe(
        doc_df,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
    )
    selected_row = event.selection.rows[0] if event.selection.rows else 0
    selected_doc = docs[selected_row]

    with st.expander("Document Preview", expanded=True):
        st.markdown(f"**{selected_doc.topic or selected_doc.document_id}**")
        if selected_doc.description:
            st.caption(selected_doc.description)
        st.write(selected_doc.doc_text)
        json_expander("Document metadata", selected_doc.doc_metadata)

with right:
    st.subheader("QA Pairs")
    st.caption(f"`{selected_doc.document_id}`")
    query = st.text_input("Filter questions", placeholder="Search question or answer...")
    qa_df = qa_table(selected_doc)
    if query:
        mask = qa_df.apply(
            lambda row: query.lower() in " ".join(str(v).lower() for v in row.values),
            axis=1,
        )
        qa_df = qa_df[mask]

    if qa_df.empty:
        st.info("No QA pairs match the filter.")
    else:
        qa_event = st.dataframe(
            qa_df,
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
        )
        qa_index = qa_event.selection.rows[0] if qa_event.selection.rows else 0
        selected_qa_id = qa_df.iloc[qa_index]["qa_id"]
        selected_qa = next(qa for qa in selected_doc.qa_pairs if qa["qa_id"] == selected_qa_id)
        with st.expander("Full QA", expanded=True):
            st.markdown("**Question**")
            st.write(selected_qa["question"])
            st.markdown("**Ground Truth Answer**")
            st.write(selected_qa.get("answer"))
            json_expander("QA metadata", selected_qa.get("metadata") or {})
