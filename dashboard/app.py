from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.lib.data import documents_for_difficulty, load_repo_meta
from dashboard.lib.ui import page_setup


repo_id, difficulty = page_setup("Home")

meta = load_repo_meta(repo_id)
docs = documents_for_difficulty(repo_id, difficulty)

st.title("SHINE Dashboard")
st.caption("Artifact browsing, interactive LoRA evaluation, routing inspection, and result analysis.")

left, right = st.columns([2, 1])
with left:
    st.subheader(meta.get("repo_id", repo_id))
    if meta.get("commit"):
        st.write(f"Commit: `{meta['commit']}`")
    st.write(
        "Use the pages in the Streamlit sidebar to inspect generated documents, "
        "ask side-by-side evaluation questions, route questions to LoRAs, and "
        "review existing logged results."
    )

with right:
    st.metric("Documents", len(docs))
    st.metric("Difficulty", difficulty)

st.info(
    "Interactive model calls require the repo's `config/eval/*.yaml` model paths, "
    "LoRA files, and judge API environment to be available on this machine. "
    "The artifact and results pages work from local JSON/JSONL files only."
)
