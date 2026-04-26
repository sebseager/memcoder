"""Transformers+PEFT fallback server with the same HTTP surface as vLLM's
LoRA endpoints. Use this when vllm-metal's LoRA path isn't viable (or for
parity testing).

Endpoints:
    GET  /v1/models                  — list loaded adapters
    POST /v1/load_lora_adapter       — {lora_name, lora_path}; re-runs the
                                       hypernetwork on source_doc.txt from
                                       the adapter dir (written by make_lora.py).
    POST /v1/unload_lora_adapter     — {lora_name}
    POST /v1/chat/completions        — OpenAI-shaped; `model` selects the adapter.
                                       Use the base model id to run without any
                                       adapter.

Run (from memcoder/experiments/):
    uv run uvicorn serving.fallback_server:app --host 0.0.0.0 --port 8001

Configuration via env vars:
    D2L_CHECKPOINT_PATH  path to hypernetwork checkpoint .bin
                         (default: doc-to-lora/trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin)
    DEVICE               cuda|mps|cpu (auto if unset)
    DTYPE                bfloat16|float16|float32 (default bfloat16)
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
MEMCODER_ROOT = EXPERIMENTS_DIR.parent
VENDOR_SRC = MEMCODER_ROOT / "vendor" / "doc-to-lora" / "src"
sys.path.insert(0, str(VENDOR_SRC))

from ctx_to_lora.model_loading import get_tokenizer  # noqa: E402
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel  # noqa: E402


DEFAULT_CKPT = (
    EXPERIMENTS_DIR / "doc-to-lora/trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
)


def _pick_device() -> torch.device:
    d = os.environ.get("DEVICE")
    if d:
        return torch.device(d)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ServerState:
    def __init__(self) -> None:
        ckpt_path = Path(os.environ.get("D2L_CHECKPOINT_PATH", str(DEFAULT_CKPT)))
        dtype_name = os.environ.get("DTYPE", "bfloat16")
        self.dtype = getattr(torch, dtype_name)
        self.device = _pick_device()

        print(f"[fallback] loading hypernetwork from {ckpt_path} on {self.device} ({dtype_name})")
        state_dict = torch.load(str(ckpt_path), weights_only=False, map_location="cpu")
        model = ModulatedPretrainedModel.from_state_dict(
            state_dict, train=False, use_flash_attn=False, use_sequence_packing=False,
        )
        self.model = model.to(self.device).to(self.dtype)
        self.model.eval()

        base_name = self.model.base_model.config.name_or_path
        self.base_model_id = base_name
        self.base_tokenizer = get_tokenizer(base_name, train=False)
        self.adapters: dict[str, dict[str, Any]] = {}
        self.lock = Lock()  # only one generation at a time (single-model state)


STATE: ServerState | None = None
app = FastAPI(title="Doc-to-LoRA fallback server", version="0.1")


@app.on_event("startup")
def _startup() -> None:
    global STATE
    STATE = ServerState()


class LoadAdapterReq(BaseModel):
    lora_name: str
    lora_path: str


class UnloadAdapterReq(BaseModel):
    lora_name: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatReq(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.0


@app.get("/v1/models")
def list_models() -> dict:
    assert STATE is not None
    data = [{"id": STATE.base_model_id, "object": "model"}]
    data += [{"id": n, "object": "model", "parent": STATE.base_model_id} for n in STATE.adapters]
    return {"object": "list", "data": data}


@app.post("/v1/load_lora_adapter")
def load_lora_adapter(req: LoadAdapterReq) -> dict:
    assert STATE is not None
    adapter_dir = Path(req.lora_path)
    doc_path = adapter_dir / "source_doc.txt"
    if not doc_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"source_doc.txt not found in {adapter_dir}. The fallback "
                   f"server reads the source doc (written by make_lora.py) and "
                   f"re-runs the hypernetwork instead of loading PEFT weights.",
        )
    doc = doc_path.read_text(encoding="utf-8")

    with STATE.lock, torch.inference_mode():
        STATE.model.reset()
        STATE.model.internalize(doc)
        STATE.adapters[req.lora_name] = dict(STATE.model.generated_loras)

    return {"lora_name": req.lora_name, "status": "loaded"}


@app.post("/v1/unload_lora_adapter")
def unload_lora_adapter(req: UnloadAdapterReq) -> dict:
    assert STATE is not None
    STATE.adapters.pop(req.lora_name, None)
    return {"lora_name": req.lora_name, "status": "unloaded"}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatReq) -> dict:
    assert STATE is not None
    use_adapter = req.model != STATE.base_model_id
    if use_adapter and req.model not in STATE.adapters:
        raise HTTPException(status_code=404, detail=f"Unknown adapter: {req.model}")

    chat = [m.model_dump() for m in req.messages]
    input_ids = STATE.base_tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_tensors="pt",
    ).to(STATE.device)

    with STATE.lock, torch.inference_mode():
        if use_adapter:
            STATE.model.generated_loras = STATE.adapters[req.model]
        else:
            STATE.model.reset()

        do_sample = req.temperature > 0
        outputs = STATE.model.generate(
            input_ids=input_ids,
            max_new_tokens=req.max_tokens,
            do_sample=do_sample,
            temperature=req.temperature if do_sample else 1.0,
        )
        text = STATE.base_tokenizer.decode(
            outputs[0][input_ids.shape[1]:], skip_special_tokens=True,
        )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }
