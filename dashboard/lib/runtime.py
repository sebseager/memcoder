from __future__ import annotations

import asyncio
import os
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from dashboard.lib.data import PROJECT_ROOT, load_ledger
from eval.config import JudgeConfig, load_run_config


DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


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


def start_lora_load(repo_id: str, lora_id: str, lora_path: str | None) -> None:
    if not lora_path:
        return
    key = (repo_id, lora_id, lora_path)
    if st.session_state.get("lora_loading_key") == key:
        return
    config_path = config_for_repo(repo_id)
    if config_path is None:
        st.session_state["lora_load_error"] = "No eval config found for this repo."
        return
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
    try:
        future.result()
    except Exception as exc:  # noqa: BLE001
        st.session_state["loaded_lora_id"] = None
        st.session_state["lora_load_error"] = str(exc)
    else:
        st.session_state["loaded_lora_id"] = lora_id
        st.session_state["lora_load_error"] = None


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
        "metamodel": metamodel,
        "tokenizer": tokenizer,
    }


@st.cache_resource(show_spinner=False)
def load_lora_resource(
    repo_id: str,
    lora_id: str,
    lora_path: str,
    config_path: str,
) -> Any:
    del repo_id, lora_id
    runtime = load_model_runtime(config_path)
    return runtime["model_module"].load_lora_dict(Path(lora_path), runtime["qwen_device"])


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
        lora_dict = load_lora_resource(repo_id, lora_id, lora_path, config_path)
    messages = model_module.build_messages(question, doc_for_prompt, condition=condition)
    return model_module.generate_answer(
        metamodel=runtime["metamodel"],
        tokenizer=runtime["tokenizer"],
        qwen_device=runtime["qwen_device"],
        messages=messages,
        lora_dict=lora_dict,
        max_new_tokens=runtime["cfg"].model.max_new_tokens,
        conversation_max_length=runtime["cfg"].model.conversation_max_length,
    )


@st.cache_data(show_spinner=False)
def judge_answer_cached(
    repo_id: str,
    question: str,
    answer: str,
    expected_answer: str,
) -> dict[str, Any]:
    config_path = config_for_repo(repo_id)
    if config_path is None:
        raise RuntimeError("No eval config found for this repo.")
    cfg = load_run_config(Path(config_path))
    return _judge_one(cfg.judge, question, answer, expected_answer)


def _judge_one(
    judge_cfg: JudgeConfig,
    question: str,
    answer: str,
    expected_answer: str,
) -> dict[str, Any]:
    from eval.judge import _JudgeContext, _judge_row, _load_dotenv, _load_prompt
    from openai import AsyncOpenAI

    if judge_cfg.dotenv_path is not None:
        _load_dotenv(judge_cfg.dotenv_path)
    api_key = os.environ.get(judge_cfg.api_key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {judge_cfg.api_key_env!r} is not set.")

    async def run() -> dict[str, Any]:
        ctx = _JudgeContext(
            cfg=judge_cfg,
            rubric_template=_load_prompt(judge_cfg.prompt),
            semaphore=asyncio.Semaphore(1),
            client=AsyncOpenAI(api_key=api_key),
        )
        row = {"question": question, "answer": answer, "expected_answer": expected_answer}
        return await _judge_row(ctx, row)

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(run())
    finally:
        loop.close()


@st.cache_resource(show_spinner=False)
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
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return {"tokenizer": tokenizer, "model": model, "device": device}


@st.cache_data(show_spinner=False)
def route_question_cached(
    repo_id: str,
    question: str,
    top_k: int,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cuda",
) -> dict[str, Any]:
    from scripts.embedding_router import flatten_loras, route_question

    ledger = load_ledger(repo_id)
    loras = flatten_loras(ledger, PROJECT_ROOT / "artifacts" / repo_id / "ledger.json")
    runtime = load_embedding_runtime(embedding_model, device)
    return route_question(
        question=question,
        loras=loras,
        tokenizer=runtime["tokenizer"],
        model=runtime["model"],
        device=runtime["device"],
        top_k=top_k,
    )
