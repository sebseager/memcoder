from __future__ import annotations

import asyncio
import gc
import os
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from dashboard.lib.data import PROJECT_ROOT, load_ledger
from eval.config import JudgeConfig, load_run_config

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EMBEDDING_DEVICE = "cpu"
DEMO_MAX_NEW_TOKENS = 128

_ANSWER_STATE_KEYS = (
    "side_by_side_answers",
    "side_by_side_judges",
    "routing_result",
    "routing_answer",
    "routing_judge",
)


@st.cache_data(show_spinner=False)
def config_for_repo(repo_id: str, routing: str | None = None) -> str | None:
    config_dir = PROJECT_ROOT / "config" / "eval"
    if not config_dir.exists():
        return None
    matches: list[Path] = []
    for path in sorted(config_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if routing and raw.get("routing") != routing:
            continue
        for selector in raw.get("artifacts") or []:
            root = selector.get("root") if isinstance(selector, dict) else None
            if root and Path(str(root)).name == repo_id:
                matches.append(path)
                break
    if not matches:
        return None
    return str(matches[0])


@st.cache_resource(show_spinner=False)
def executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="shine-dashboard")


def sync_repo_selection(repo_id: str) -> None:
    previous_repo = st.session_state.get("active_repo_id")
    if previous_repo and previous_repo != repo_id:
        release_lora_resource(reset_status=True)
        for key in _ANSWER_STATE_KEYS:
            st.session_state.pop(key, None)
    st.session_state["active_repo_id"] = repo_id


def _lora_key(repo_id: str, lora_id: str, lora_path: str) -> tuple[str, str, str]:
    return (repo_id, lora_id, lora_path)


def release_lora_resource(*, reset_status: bool = False) -> None:
    future = st.session_state.get("lora_future")
    if isinstance(future, Future) and not future.done():
        future.cancel()

    lora_dict = st.session_state.pop("loaded_lora_dict", None)
    st.session_state.pop("loaded_lora_key", None)
    st.session_state.pop("loaded_lora_config_path", None)
    if lora_dict is not None:
        try:
            from eval import model as model_module

            model_module.free_lora_dict(lora_dict)
        except Exception:  # noqa: BLE001
            del lora_dict
            gc.collect()

    if reset_status:
        for key in (
            "lora_future",
            "lora_loading_key",
            "pending_lora_id",
            "loaded_lora_id",
            "lora_load_error",
        ):
            st.session_state.pop(key, None)


def _store_lora_resource(
    *,
    key: tuple[str, str, str],
    config_path: str,
    lora_dict: Any,
) -> None:
    if st.session_state.get("loaded_lora_key") != key:
        release_lora_resource(reset_status=False)
    st.session_state["loaded_lora_key"] = key
    st.session_state["loaded_lora_config_path"] = config_path
    st.session_state["loaded_lora_dict"] = lora_dict
    st.session_state["loaded_lora_id"] = key[1]
    st.session_state["lora_load_error"] = None


def start_lora_load(repo_id: str, lora_id: str, lora_path: str | None) -> None:
    if not lora_path:
        return
    key = _lora_key(repo_id, lora_id, lora_path)
    if st.session_state.get("loaded_lora_key") == key:
        st.session_state["loaded_lora_id"] = lora_id
        return
    if st.session_state.get("lora_loading_key") == key:
        return
    current_future = st.session_state.get("lora_future")
    if isinstance(current_future, Future) and not current_future.done():
        current_future.cancel()
        if not current_future.cancelled():
            st.session_state["lora_load_error"] = (
                "Waiting for the current LoRA load to finish before starting another."
            )
            return
    config_path = config_for_repo(repo_id)
    if config_path is None:
        st.session_state["lora_load_error"] = "No eval config found for this repo."
        return
    release_lora_resource(reset_status=False)
    st.session_state["lora_loading_key"] = key
    st.session_state["pending_lora_id"] = lora_id
    st.session_state["lora_load_error"] = None
    st.session_state["lora_future"] = executor().submit(
        load_lora_resource,
        repo_id,
        lora_id,
        lora_path,
        config_path,
    )


def update_lora_load_status() -> None:
    future = st.session_state.get("lora_future")
    if not isinstance(future, Future) or not future.done():
        return
    lora_id = st.session_state.get("pending_lora_id")
    key = st.session_state.get("lora_loading_key")
    config_path = config_for_repo(str(key[0])) if isinstance(key, tuple) else None
    try:
        lora_dict = future.result()
    except CancelledError:
        st.session_state["loaded_lora_id"] = None
        st.session_state["lora_load_error"] = None
    except Exception as exc:  # noqa: BLE001
        st.session_state["loaded_lora_id"] = None
        st.session_state["lora_load_error"] = str(exc)
    else:
        if isinstance(key, tuple) and config_path is not None:
            _store_lora_resource(key=key, config_path=config_path, lora_dict=lora_dict)
        st.session_state["loaded_lora_id"] = lora_id
        st.session_state["lora_load_error"] = None
    finally:
        st.session_state.pop("lora_future", None)
        st.session_state.pop("lora_loading_key", None)
        st.session_state.pop("pending_lora_id", None)


def set_embedding_model_status(
    state: str,
    *,
    model_name: str | None = None,
    device: str | None = None,
    error: str | None = None,
) -> None:
    st.session_state["embedding_model_status"] = {
        "state": state,
        "model_name": model_name,
        "device": device,
        "error": error,
    }


@st.cache_resource(show_spinner=False)
def load_model_runtime(config_path: str) -> dict[str, Any]:
    from eval import model as model_module

    cfg = load_run_config(Path(config_path))
    if not cfg.model.qwen_base.exists():
        raise FileNotFoundError(
            f"Qwen base model path does not exist: {cfg.model.qwen_base}. "
            "Update the matching config/eval YAML for this machine."
        )
    model_module.ensure_runtime()
    model_module.setup_shine_imports(cfg.model.shine_root)
    qwen_device = model_module.resolve_qwen_device(cfg.model.qwen_cuda)
    metamodel, tokenizer = model_module.load_qwen_runtime(cfg.model, qwen_device)
    return {
        "cfg": cfg,
        "model_module": model_module,
        "qwen_device": qwen_device,
        "qwen_dtype": model_module.preferred_model_dtype(qwen_device),
        "metamodel": metamodel,
        "tokenizer": tokenizer,
    }


def load_lora_resource(
    repo_id: str,
    lora_id: str,
    lora_path: str,
    config_path: str,
) -> Any:
    del repo_id, lora_id
    runtime = load_model_runtime(config_path)
    return runtime["model_module"].load_lora_dict(
        Path(lora_path),
        runtime["qwen_device"],
        dtype=runtime["qwen_dtype"],
    )


def get_lora_resource(repo_id: str, lora_id: str, lora_path: str, config_path: str) -> Any:
    key = _lora_key(repo_id, lora_id, lora_path)
    if st.session_state.get("loaded_lora_key") == key:
        return st.session_state.get("loaded_lora_dict")
    lora_dict = load_lora_resource(repo_id, lora_id, lora_path, config_path)
    _store_lora_resource(key=key, config_path=config_path, lora_dict=lora_dict)
    return lora_dict


@st.cache_data(show_spinner=False)
def generate_answer_cached(
    repo_id: str,
    condition: str,
    question: str,
    document_text: str,
    lora_id: str | None,
    lora_path: str | None,
) -> dict[str, str]:
    config_path = config_for_repo(repo_id)
    if config_path is None:
        raise RuntimeError("No eval config found for this repo.")
    runtime = load_model_runtime(config_path)
    model_module = runtime["model_module"]
    doc_for_prompt = document_text if condition == "in_context" else None
    lora_dict = None
    if condition == "shine":
        if not lora_id or not lora_path:
            raise RuntimeError("SHINE generation requires a selected LoRA.")
        lora_dict = get_lora_resource(repo_id, lora_id, lora_path, config_path)
    messages = model_module.build_messages(question, doc_for_prompt, condition=condition)
    return model_module.generate_answer(
        metamodel=runtime["metamodel"],
        tokenizer=runtime["tokenizer"],
        qwen_device=runtime["qwen_device"],
        messages=messages,
        lora_dict=lora_dict,
        max_new_tokens=min(runtime["cfg"].model.max_new_tokens, DEMO_MAX_NEW_TOKENS),
        conversation_max_length=runtime["cfg"].model.conversation_max_length,
    )


@st.cache_data(show_spinner=False)
def judge_answer_cached(
    repo_id: str,
    question: str,
    answer: str,
    expected_answer: str,
    document_id: str | None = None,
    document_text: str | None = None,
) -> dict[str, Any]:
    config_path = config_for_repo(repo_id)
    if config_path is None:
        raise RuntimeError("No eval config found for this repo.")
    cfg = load_run_config(Path(config_path))
    return _judge_one(
        cfg.judge,
        repo_id=repo_id,
        question=question,
        answer=answer,
        expected_answer=expected_answer,
        document_id=document_id,
        document_text=document_text,
    )


def _judge_one(
    judge_cfg: JudgeConfig,
    *,
    repo_id: str,
    question: str,
    answer: str,
    expected_answer: str,
    document_id: str | None,
    document_text: str | None,
) -> dict[str, Any]:
    from eval.judge import _JudgeContext, _judge_row, _load_dotenv, _load_prompt
    from openai import AsyncOpenAI

    if judge_cfg.dotenv_path is not None:
        _load_dotenv(judge_cfg.dotenv_path)
    api_key = os.environ.get(judge_cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {judge_cfg.api_key_env!r} is not set.")

    async def run() -> dict[str, Any]:
        rubric_template, needs_doc = _load_prompt(judge_cfg.prompt)
        docs_by_key = {}
        if needs_doc and document_id and document_text:
            docs_by_key[(repo_id, document_id)] = document_text
        ctx = _JudgeContext(
            cfg=judge_cfg,
            rubric_template=rubric_template,
            needs_doc=needs_doc,
            docs_by_key=docs_by_key,
            semaphore=asyncio.Semaphore(1),
            client=AsyncOpenAI(api_key=api_key),
        )
        row = {
            "repo_id": repo_id,
            "document_id": document_id or "",
            "question": question,
            "answer": answer,
            "expected_answer": expected_answer,
        }
        return await _judge_row(ctx, row)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(run())
    finally:
        loop.close()


@st.cache_resource(show_spinner=False, max_entries=2)
def load_embedding_runtime(model_name: str, requested_device: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = requested_device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return {"tokenizer": tokenizer, "model": model, "device": device}


@st.cache_data(show_spinner=False)
def lora_embeddings_cached(
    repo_id: str,
    embedding_model: str,
    device: str,
) -> dict[str, Any]:
    from scripts.embedding_router import embed_texts, flatten_loras, make_lora_routing_text

    ledger = load_ledger(repo_id)
    loras = flatten_loras(ledger, PROJECT_ROOT / "artifacts" / repo_id / "ledger.json")
    runtime = load_embedding_runtime(embedding_model, device)
    routing_texts = [make_lora_routing_text(lora) for lora in loras]
    embeddings = embed_texts(
        routing_texts,
        runtime["tokenizer"],
        runtime["model"],
        runtime["device"],
    )
    return {"loras": loras, "embeddings": embeddings}


@st.cache_data(show_spinner=False)
def route_question_cached(
    repo_id: str,
    question: str,
    top_k: int,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    device: str = DEFAULT_EMBEDDING_DEVICE,
) -> dict[str, Any]:
    from scripts.embedding_router import cosine_scores, embed_texts

    runtime = load_embedding_runtime(embedding_model, device)
    cached = lora_embeddings_cached(repo_id, embedding_model, runtime["device"])
    loras = cached["loras"]
    lora_embeddings = cached["embeddings"]
    question_embedding = embed_texts(
        [question],
        tokenizer=runtime["tokenizer"],
        model=runtime["model"],
        device=runtime["device"],
    )
    scores = cosine_scores(question_embedding, lora_embeddings)
    ranked = sorted(
        [
            {
                "rank": i + 1,
                "lora_id": loras[idx].get("lora_id"),
                "score": scores[idx],
                "topic": loras[idx].get("topic"),
                "source_doc": loras[idx].get("source_doc"),
                "lora_path": loras[idx].get("lora_path"),
            }
            for i, idx in enumerate(sorted(range(len(scores)), key=lambda j: scores[j], reverse=True))
        ],
        key=lambda x: x["rank"],
    )
    selected = ranked[: max(1, top_k)]
    return {
        "question": question,
        "method": "embedding_cosine_v1",
        "embedding_model": embedding_model,
        "embedding_device": runtime["device"],
        "top_k": top_k,
        "selected_lora_ids": [r["lora_id"] for r in selected],
        "ranked_loras": ranked,
    }
