"""Generate a LoRA adapter from a document via the Doc-to-LoRA hypernetwork
and hot-load it into a running vLLM server.

Usage:
    uv run python serving/make_lora.py \\
        --doc path/to/repo_qa.txt --name repo-foo --smoke-test

Run from memcoder/experiments/ (or pass absolute paths).
Imports the hypernetwork code from memcoder/vendor/doc-to-lora (git submodule).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests
import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent           # memcoder/experiments/
MEMCODER_ROOT = EXPERIMENTS_DIR.parent                              # memcoder/
VENDOR_SRC = MEMCODER_ROOT / "vendor" / "doc-to-lora" / "src"
sys.path.insert(0, str(VENDOR_SRC))

from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel  # noqa: E402


ATTN_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"}
MLP_MODULES = {"down_proj", "up_proj", "gate_proj"}
DEFAULT_CKPT = (
    EXPERIMENTS_DIR / "doc-to-lora/trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
)
DEFAULT_ADAPTERS_DIR = EXPERIMENTS_DIR / "serving/adapters"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--doc", required=True, type=Path, help="Document text file")
    p.add_argument("--name", required=True, help="Adapter name (used in vLLM `model` field)")
    p.add_argument(
        "--ckpt",
        type=Path,
        default=Path(os.environ.get("D2L_CHECKPOINT_PATH") or DEFAULT_CKPT),
        help=f"Hypernetwork checkpoint .bin (default: {DEFAULT_CKPT})",
    )
    p.add_argument("--server", default="http://localhost:8000", help="vLLM base URL")
    p.add_argument("--adapters-dir", type=Path, default=DEFAULT_ADAPTERS_DIR)
    p.add_argument("--device", default=None, help="cuda|mps|cpu (auto if omitted)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--smoke-test", action="store_true", help="Chat the new adapter with a short prompt")
    p.add_argument("--smoke-prompt", default="Summarize the context in one sentence.")
    return p.parse_args()


def pick_device(flag: str | None) -> torch.device:
    if flag:
        return torch.device(flag)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_hypernet(ckpt_path: Path, device: torch.device, dtype: torch.dtype) -> ModulatedPretrainedModel:
    state_dict = torch.load(str(ckpt_path), weights_only=False, map_location="cpu")
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict,
        train=False,
        use_flash_attn=False,
        use_sequence_packing=False,
    )
    model = model.to(device).to(dtype)
    model.eval()
    return model


def bake_generated_into_peft(model: ModulatedPretrainedModel) -> None:
    """Copy `model.generated_loras` into the base PEFT adapter's parameters so
    `model.base_model.save_pretrained(...)` emits standard PEFT files."""
    if model.generated_loras is None:
        raise RuntimeError("No generated_loras on model — call model.internalize(doc) first.")

    layer_indices = [int(i) for i in model.hypernet.layer_indices]
    named = dict(model.base_model.named_modules())

    for prefix in ("base_model.model.model.layers", "base_model.model.layers"):
        if f"{prefix}.{layer_indices[0]}" in named:
            base_prefix = prefix
            break
    else:
        raise RuntimeError("Could not locate transformer layers in the PEFT model tree.")

    with torch.no_grad():
        for tm, AB in model.generated_loras.items():
            sub = "self_attn" if tm in ATTN_MODULES else ("mlp" if tm in MLP_MODULES else None)
            if sub is None:
                raise RuntimeError(f"Unknown target module family: {tm!r}")
            A = AB["A"][0]  # [n_layers_active, r, d_in]
            B = AB["B"][0]  # [n_layers_active, r, d_out]
            for i, L in enumerate(layer_indices):
                key = f"{base_prefix}.{L}.{sub}.{tm}"
                lora_A_mod = named.get(f"{key}.lora_A.default")
                lora_B_mod = named.get(f"{key}.lora_B.default")
                if lora_A_mod is None or lora_B_mod is None:
                    raise RuntimeError(f"Missing PEFT submodule for {key}")
                lora_A_mod.weight.data.copy_(A[i].to(lora_A_mod.weight.dtype))
                # generated B is [r, d_out]; PEFT Linear B weight is [d_out, r]
                lora_B_mod.weight.data.copy_(B[i].T.contiguous().to(lora_B_mod.weight.dtype))


def save_adapter(model: ModulatedPretrainedModel, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.base_model.save_pretrained(str(out_dir))
    return out_dir.resolve()


def load_adapter_into_server(server: str, name: str, path: str) -> dict:
    url = f"{server.rstrip('/')}/v1/load_lora_adapter"
    payload = {"lora_name": name, "lora_path": path}
    r = requests.post(url, json=payload, timeout=120)
    if r.status_code >= 400 and "already" in r.text.lower():
        payload["load_inplace"] = True
        r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    try:
        return r.json()
    except ValueError:
        return {"status": r.status_code, "text": r.text}


def smoke_test(server: str, name: str, prompt: str) -> dict:
    url = f"{server.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.0,
    }
    r = requests.post(url, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    device = pick_device(args.device)
    doc_text = args.doc.read_text(encoding="utf-8")

    print(f"[1/5] Loading hypernetwork from {args.ckpt} on {device} ({dtype})...")
    model = load_hypernet(args.ckpt, device, dtype)

    print(f"[2/5] Internalizing doc ({len(doc_text):,} chars)...")
    model.internalize(doc_text)

    print("[3/5] Baking generated LoRA weights into the PEFT adapter...")
    bake_generated_into_peft(model)

    out_dir = args.adapters_dir / args.name
    print(f"[4/5] Saving PEFT adapter to {out_dir}...")
    abs_path = str(save_adapter(model, out_dir))
    # Keep the source doc next to the PEFT files so the fallback server can re-internalize.
    (out_dir / "source_doc.txt").write_text(doc_text, encoding="utf-8")

    print(f"[5/5] Hot-loading adapter '{args.name}' into vLLM at {args.server}...")
    resp = load_adapter_into_server(args.server, args.name, abs_path)
    print(f"      server: {resp}")

    if args.smoke_test:
        print(f"\n--- smoke test (prompt: {args.smoke_prompt!r}) ---")
        result = smoke_test(args.server, args.name, args.smoke_prompt)
        msg = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        print(msg or json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
