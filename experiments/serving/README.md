# Serving Qwen3-4B with hot-swappable Doc-to-LoRA adapters

Three entrypoints. Run everything from `memcoder/experiments/`.

The hypernetwork code is imported from the `memcoder/vendor/doc-to-lora` git
submodule; no need to install that repo separately.

## One-time setup

```sh
bash serving/setup.sh
```

Installs `vllm-metal`, `fastapi`, `uvicorn`, `requests`. Downloads the Sakana
Qwen3-4B hypernetwork to `doc-to-lora/trained_d2l/qwen_4b_d2l/` (the same
layout the sibling `doc-to-lora/demo_*.py` scripts already use).

## Main path: vLLM with runtime LoRA

Terminal A — start the server:

```sh
bash serving/serve.sh
```

Serves `Qwen/Qwen3-4B-Instruct-2507` on `http://localhost:8000` with
`--enable-lora --max-loras 4 --max-lora-rank 8` and
`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`. Override via env vars, e.g.
`PORT=9000 MAX_LORAS=8 bash serving/serve.sh`.

Terminal B — generate an adapter from a doc and hot-load it:

```sh
uv run python serving/make_lora.py \
    --doc doc-to-lora/data/sakana_wiki.txt \
    --name sakana \
    --smoke-test
```

What it does: loads the hypernetwork → `model.internalize(doc)` → bakes the
generated A/B weights into a PEFT adapter → saves to
`serving/adapters/<name>/` → `POST /v1/load_lora_adapter`. With
`--smoke-test` it also runs a short chat completion against the new adapter.

Inference from any OpenAI client — route per request by setting `model`:

```sh
curl http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "sakana",
      "messages": [{"role": "user", "content": "Tell me about Sakana AI."}],
      "max_tokens": 256
    }'
```

Re-running `make_lora.py --name sakana` with a different doc replaces the
adapter in place — the server keeps running.

## Fallback path: transformers + PEFT via FastAPI

Use when the vllm-metal LoRA path doesn't work on your Mac.

```sh
uv run uvicorn serving.fallback_server:app --host 0.0.0.0 --port 8001
```

Same HTTP shape as vLLM: `GET /v1/models`, `POST /v1/load_lora_adapter`,
`POST /v1/unload_lora_adapter`, `POST /v1/chat/completions`. Point
`make_lora.py` at it:

```sh
uv run python serving/make_lora.py \
    --doc doc-to-lora/data/sakana_wiki.txt --name sakana \
    --server http://localhost:8001 --smoke-test
```

The fallback server reads `source_doc.txt` from the adapter dir (written by
`make_lora.py`) and re-runs the hypernetwork internally rather than loading
PEFT weight files. One generation at a time (lock-protected).

## Files

| File | Purpose |
| --- | --- |
| `setup.sh` | Install deps + download hypernetwork checkpoint |
| `serve.sh` | Launch vLLM with runtime LoRA support |
| `make_lora.py` | doc → hypernetwork → PEFT dir → POST `load_lora_adapter` |
| `fallback_server.py` | transformers+PEFT HTTP server, same API shape |
| `adapters/` | Generated PEFT adapters (one subdir per name) |

## Notes / gotchas

- `--max-lora-rank 8` matches the Doc-to-LoRA paper's hypernetwork output.
  If a different rank is ever used, bump this flag or vLLM will reject the
  adapter.
- On 16 GB unified memory, set `MAX_MODEL_LEN=4096` to keep the KV cache
  manageable.
- Set `D2L_CHECKPOINT_PATH` to override the default checkpoint location
  (same env var the sibling `doc-to-lora/` demo scripts use).
- `vllm-metal` requires Python ≥ 3.12 but the experiments env is pinned to
  3.10 (for the hypernetwork + flash-attn). `serve.sh` handles this by
  launching vLLM via `uv run --isolated --python 3.12 --with vllm-metal`,
  which uv manages as a separate ephemeral env. `make_lora.py` and
  `fallback_server.py` still run in the 3.10 env and talk to vLLM over HTTP.
