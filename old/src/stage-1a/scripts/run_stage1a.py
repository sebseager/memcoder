#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import gc
import json
import logging
import re
import subprocess
import sys
import time
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from omegaconf import OmegaConf


@dataclass
class SliceArtifact:
    context_text: str
    covered_refs: list[str]
    missing_refs: list[str]
    coverage_fraction: float
    token_count: int


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    stage1a_root = repo_root / "src" / "stage-1a"
    default_instances = repo_root / "src" / "stage-0" / "outputs" / "stage1_instances.jsonl"
    default_out = stage1a_root / "outputs" / "stage1a_predictions.jsonl"
    default_meta = stage1a_root / "outputs" / "stage1a_run_meta.json"
    default_base_model = stage1a_root / "models" / "Qwen3-8B"
    default_ckpt = (
        stage1a_root
        / "checkpoints"
        / "8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150"
        / "iftpwc"
        / "checkpoint-epoch-2"
    )
    default_vendor = repo_root / "vendor" / "SHINE"

    p = argparse.ArgumentParser(
        description=(
            "Run Stage 1a SHINE zero-shot generation on Stage 0 instances "
            "(HPC-friendly: no docker/harness calls)."
        )
    )
    p.add_argument("--instances-jsonl", type=Path, default=default_instances)
    p.add_argument("--output-jsonl", type=Path, default=default_out)
    p.add_argument("--output-meta-json", type=Path, default=default_meta)
    p.add_argument("--max-instances", type=int, default=20)
    p.add_argument("--instance-ids-file", type=Path, default=None)
    p.add_argument("--base-model-path", type=Path, default=default_base_model)
    p.add_argument("--tokenizer-path", type=Path, default=None)
    p.add_argument("--checkpoint-dir", type=Path, default=default_ckpt)
    p.add_argument("--vendor-shine-dir", type=Path, default=default_vendor)
    p.add_argument("--repo-cache-dir", type=Path, default=stage1a_root / "cache" / "repos")
    p.add_argument("--context-max-tokens", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-conversation-length", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug-every", type=int, default=1)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_text(path: str | None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def ensure_repo_checkout(repo: str, commit: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_key = re.sub(r"[^A-Za-z0-9_.-]", "__", repo)
    repo_dir = cache_dir / repo_key
    url = f"https://github.com/{repo}.git"

    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "--no-tags", "--filter=blob:none", url, str(repo_dir)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    subprocess.run(
        ["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", commit],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "checkout", "--force", commit],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return repo_dir


def extract_function_source_from_full_file(
    full_file: str,
    function_name: str,
    start_line: int,
) -> str:
    if not full_file.strip():
        return ""
    try:
        tree = ast.parse(full_file)
    except SyntaxError:
        return ""
    target: ast.AST | None = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != function_name:
            continue
        if int(getattr(node, "lineno", -1)) == start_line:
            target = node
            break
        if target is None:
            target = node
    if target is None:
        return ""
    return node_source_segment(full_file, target)


def function_body_from_source(function_source: str, function_name: str) -> str:
    if not function_source.strip():
        return ""
    try:
        tree = ast.parse(textwrap.dedent(function_source))
    except SyntaxError:
        return ""
    target: ast.AST | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            target = node
            break
    if target is None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                target = node
                break
    if target is None or not getattr(target, "body", None):
        return ""
    lines = function_source.splitlines()
    start = int(target.body[0].lineno)
    end = int(getattr(target, "end_lineno", start))
    return "\n".join(lines[start - 1 : end]).rstrip()


def resolve_instances(rows: list[dict[str, Any]], repo_cache_dir: Path, log: logging.Logger) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        rec = dict(row)
        if not rec.get("full_file"):
            rec["full_file"] = read_text(rec.get("full_file_artifact"))
        if not rec.get("ground_truth_body"):
            rec["ground_truth_body"] = read_text(rec.get("ground_truth_artifact"))
        if not rec.get("function_source"):
            rec["function_source"] = read_text(rec.get("function_source_artifact"))

        if not rec.get("full_file"):
            repo = str(rec.get("repo", ""))
            base_commit = str(rec.get("base_commit", ""))
            file_path = str(rec.get("file_path", ""))
            if repo and base_commit and file_path:
                try:
                    repo_dir = ensure_repo_checkout(repo=repo, commit=base_commit, cache_dir=repo_cache_dir)
                    abs_file = repo_dir / file_path
                    if abs_file.exists():
                        rec["full_file"] = abs_file.read_text(encoding="utf-8")
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "Fallback repo hydrate failed for %s (%s): %s",
                        rec.get("instance_id", ""),
                        repo,
                        exc,
                    )

        if not rec.get("function_source"):
            rec["function_source"] = extract_function_source_from_full_file(
                full_file=str(rec.get("full_file", "")),
                function_name=str(rec.get("function_name", "")),
                start_line=int(rec.get("start_line", 1)),
            )

        if not rec.get("ground_truth_body"):
            rec["ground_truth_body"] = function_body_from_source(
                function_source=str(rec.get("function_source", "")),
                function_name=str(rec.get("function_name", "")),
            )

        if not rec.get("masked_function"):
            rec["masked_function"] = masked_function_from_source(
                rec.get("function_source", ""),
                str(rec.get("function_name", "")),
            )
        out.append(rec)
    return out


def masked_function_from_source(function_source: str, function_name: str) -> str:
    if not function_source.strip():
        return ""
    try:
        tree = ast.parse(textwrap.dedent(function_source))
    except SyntaxError:
        return ""
    target: ast.AST | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            target = node
            break
    if target is None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                target = node
                break
    if target is None or not getattr(target, "body", None):
        return ""
    lines = function_source.splitlines()
    body_start_line = int(target.body[0].lineno)
    sig_lines = lines[int(target.lineno) - 1 : body_start_line - 1]
    if not sig_lines:
        return ""
    body_indent = "    "
    if body_start_line - 1 < len(lines):
        match = re.match(r"^\s*", lines[body_start_line - 1])
        if match:
            body_indent = match.group(0) or "    "
    return "\n".join(sig_lines).rstrip() + "\n" + body_indent + "pass"


def maybe_filter_instance_ids(
    rows: list[dict[str, Any]],
    ids_file: Path | None,
    max_instances: int,
) -> list[dict[str, Any]]:
    selected = rows
    if ids_file is not None:
        wanted = {
            line.strip()
            for line in ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        selected = [r for r in rows if str(r.get("instance_id", "")) in wanted]
    if max_instances > 0:
        selected = selected[:max_instances]
    return selected


def attach_parents(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            setattr(child, "_parent", node)


def node_source_segment(text: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(text, node)
    if segment is not None:
        return segment
    lines = text.splitlines()
    start = int(getattr(node, "lineno", 1))
    end = int(getattr(node, "end_lineno", start))
    return "\n".join(lines[start - 1 : end])


def collect_called_names(fn_node: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name):
            names.add(fn.id)
        elif isinstance(fn, ast.Attribute):
            names.add(fn.attr)
    return names


def collect_attribute_names(fn_node: ast.AST) -> set[str]:
    attrs: set[str] = set()
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Attribute):
            attrs.add(node.attr)
    return attrs


def module_level_defs(module: ast.Module) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out[node.target.id] = node
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out[alias.asname or alias.name.split(".")[0]] = node
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                out[alias.asname or alias.name] = node
    return out


def class_attr_defs(module: ast.Module) -> dict[str, dict[str, ast.AST]]:
    out: dict[str, dict[str, ast.AST]] = {}
    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        attrs: dict[str, ast.AST] = {}
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        attrs[target.id] = stmt
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                attrs[stmt.target.id] = stmt
        out[node.name] = attrs
    return out


def target_function_node(module: ast.Module, start_line: int, name: str) -> ast.AST | None:
    best: ast.AST | None = None
    for node in ast.walk(module):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != name:
            continue
        if int(getattr(node, "lineno", -1)) == start_line:
            return node
        if best is None:
            best = node
    return best


def function_signature_plus_doc(function_source: str, function_name: str) -> str:
    if not function_source.strip():
        return f"def {function_name}(...):"
    try:
        tree = ast.parse(textwrap.dedent(function_source))
    except SyntaxError:
        return function_source.strip()
    target: ast.AST | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            target = node
            break
    if target is None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                target = node
                break
    if target is None or not getattr(target, "body", None):
        return function_source.strip()
    lines = function_source.splitlines()
    first_body = int(target.body[0].lineno)
    sig = "\n".join(lines[int(target.lineno) - 1 : first_body - 1]).rstrip()
    doc = ast.get_docstring(target, clean=False) or ""
    if not doc:
        return sig
    doc_clean = doc.strip("\n")
    return f'{sig}\n    """{doc_clean}"""'


def build_slice(
    row: dict[str, Any],
    tokenizer,
    max_tokens: int,
) -> SliceArtifact:
    full_file = str(row.get("full_file", ""))
    refs = [str(x) for x in (row.get("external_references") or [])]
    fn_name = str(row.get("function_name", ""))
    start_line = int(row.get("start_line", 1))
    fn_source = str(row.get("function_source", ""))

    if not full_file.strip():
        return SliceArtifact(
            context_text="",
            covered_refs=[],
            missing_refs=refs,
            coverage_fraction=0.0 if refs else 1.0,
            token_count=0,
        )

    try:
        tree = ast.parse(full_file)
    except SyntaxError:
        text = fn_source[:4000]
        tok = len(tokenizer.encode(text, add_special_tokens=False))
        return SliceArtifact(
            context_text=text,
            covered_refs=[],
            missing_refs=refs,
            coverage_fraction=0.0 if refs else 1.0,
            token_count=tok,
        )

    attach_parents(tree)
    target = target_function_node(tree, start_line=start_line, name=fn_name)
    if target is None:
        text = fn_source[:4000]
        tok = len(tokenizer.encode(text, add_special_tokens=False))
        return SliceArtifact(
            context_text=text,
            covered_refs=[],
            missing_refs=refs,
            coverage_fraction=0.0 if refs else 1.0,
            token_count=tok,
        )

    defs = module_level_defs(tree)
    class_defs = class_attr_defs(tree)
    called = sorted(collect_called_names(target))
    attr_names = sorted(collect_attribute_names(target))

    sections: list[tuple[str, str, set[str]]] = []
    covered: set[str] = set()

    q_text = function_signature_plus_doc(fn_source, fn_name).strip()
    sections.append(("TARGET_SIGNATURE", q_text, set()))
    if refs:
        sections.append(
            (
                "REFERENCE_NAME_INDEX",
                "Known in-file references:\n" + ", ".join(sorted(refs)),
                set(refs),
            )
        )
        covered.update(refs)

    for name in called:
        node = defs.get(name)
        if node is None or isinstance(node, ast.ClassDef):
            continue
        sections.append(("ONE_HOP_CALLEE", node_source_segment(full_file, node), {name}))
        covered.add(name)

    for name in refs:
        if name in covered:
            continue
        node = defs.get(name)
        if node is None:
            continue
        sections.append(("REFERENCE_SYMBOL", node_source_segment(full_file, node), {name}))
        covered.add(name)

    for name in refs:
        if name not in defs:
            continue
        if not re.match(r"^[A-Z_][A-Z0-9_]*$", name):
            continue
        node = defs[name]
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            sections.append(("MODULE_CONSTANT", node_source_segment(full_file, node), {name}))
            covered.add(name)

    parent = getattr(target, "_parent", None)
    if isinstance(parent, ast.ClassDef):
        class_name = parent.name
        attr_map = class_defs.get(class_name, {})
        for attr in attr_names:
            node = attr_map.get(attr)
            if node is None:
                continue
            sections.append(("CLASS_ATTRIBUTE", node_source_segment(full_file, node), {attr}))
            covered.add(attr)

    chunked: list[str] = []
    for title, body, _names in sections:
        if not body.strip():
            continue
        chunked.append(f"# [{title}]\n{body.strip()}")

    selected = ""
    for idx, block in enumerate(chunked):
        candidate = block if idx == 0 else f"{selected}\n\n{block}"
        n_tok = len(tokenizer.encode(candidate, add_special_tokens=False))
        if n_tok <= max_tokens:
            selected = candidate
            continue
        # Keep a truncation fallback for the final overflow block.
        if not selected:
            ids = tokenizer.encode(block, add_special_tokens=False)[:max_tokens]
            selected = tokenizer.decode(ids, skip_special_tokens=True)
        break

    token_count = len(tokenizer.encode(selected, add_special_tokens=False))
    covered_in_text = [
        r
        for r in refs
        if re.search(rf"\b{re.escape(r)}\b", selected)
    ]
    missing = [r for r in refs if r not in covered_in_text]
    frac = 1.0 if not refs else len(covered_in_text) / len(refs)
    return SliceArtifact(
        context_text=selected,
        covered_refs=sorted(covered_in_text),
        missing_refs=missing,
        coverage_fraction=frac,
        token_count=token_count,
    )


def load_shine_runtime(vendor_shine_dir: Path) -> dict[str, Any]:
    if not vendor_shine_dir.exists():
        raise FileNotFoundError(f"Missing SHINE directory: {vendor_shine_dir}")
    sys.path.insert(0, str(vendor_shine_dir))
    from metanetwork_family import Metanetwork
    from utils.myfreeze import freeze
    from utils.mysaveload import load_checkpoint
    from utils.myinit import _import_class

    return {
        "Metanetwork": Metanetwork,
        "freeze": freeze,
        "load_checkpoint": load_checkpoint,
        "_import_class": _import_class,
    }


def init_metanetwork(
    runtime: dict[str, Any],
    base_model_path: Path,
    tokenizer_path: Path,
    checkpoint_dir: Path,
    device: torch.device,
) -> tuple[Any, Any]:
    Metanetwork = runtime["Metanetwork"]
    freeze = runtime["freeze"]
    load_checkpoint = runtime["load_checkpoint"]
    _import_class = runtime["_import_class"]

    conf_dict = {
        "name": "stage1a-zero-shot",
        "run": {"seed": 42, "device": "cuda"},
        "model": {
            "lora_r": 8,
            "tokenizer_from": str(tokenizer_path),
            "model_from": str(base_model_path),
            "metamodel_class_path": "LoraQwen.LoraQwen3ForCausalLM",
            "config_class_path": "LoraQwen.Qwen3Config",
        },
        "metanetwork": {
            "type": "transformer",
            "method": "rl",
            "transformer_cfg": {
                "encoder_cfg": {
                    "d_model": 4096,
                    "nhead": 32,
                    "dim_feedforward": 8192,
                    "dropout": 0.0,
                    "activation": "gelu",
                    "layer_norm_eps": 1e-5,
                    "batch_first": True,
                    "norm_first": False,
                    "bias": True,
                },
                "couple_encoder_cfg": {
                    "d_model": 4096,
                    "nhead": 32,
                    "dim_feedforward": 8192,
                    "dropout": 0.0,
                    "activation": "gelu",
                    "layer_norm_eps": 1e-5,
                    "batch_first": True,
                    "norm_first": False,
                    "bias": True,
                },
                "layer_transformer_first": True,
                "mean_pool_size": 1,
                "num_layers": 4,
                "couple_num_layers": 0,
                "scale": 0.001,
            },
        },
        "hidden_size": -1,
        "num_layers": -1,
        "num_mem_token": 4,
    }
    cfg = OmegaConf.create(conf_dict)

    MetaModelCls = _import_class(cfg.model.metamodel_class_path)
    ConfigCls = _import_class(cfg.model.config_class_path)
    config = ConfigCls.from_pretrained(cfg.model.model_from)
    config.num_mem_token = -1
    cfg.hidden_size = int(config.hidden_size)
    cfg.num_layers = int(config.num_hidden_layers)

    with torch.device("meta"):
        tmp_model = MetaModelCls(config)
    lora_params = tmp_model.lora_params_numel(cfg.model.lora_r)
    base_params = cfg.hidden_size * cfg.num_layers
    if lora_params % base_params != 0:
        raise RuntimeError(
            f"Unexpected SHINE shape mismatch: lora_params={lora_params}, "
            f"hidden*layers={base_params}"
        )
    config.num_mem_token = lora_params // base_params
    cfg.num_mem_token = int(config.num_mem_token)
    del tmp_model
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), padding_side="left", use_fast=True)
    tokenizer.add_tokens(["<RECON>", "<COMP>", "<NOTHING>"])

    metamodel = MetaModelCls.from_pretrained(str(base_model_path), config=config)
    metamodel.reset_mem_tokens()
    metamodel.resize_token_embeddings(len(tokenizer))
    metanetwork = Metanetwork(metamodel, cfg, metamodel.lora_params_numel(cfg.model.lora_r))
    metanetwork.to(device)
    freeze(metamodel)
    metanetwork.eval()

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    metanetwork, metalora, _ = load_checkpoint(metanetwork, str(checkpoint_dir), device)
    return metanetwork, metalora, tokenizer


def extract_answer(raw: str) -> str:
    text = raw.strip()
    if text.lstrip().startswith("<think>") and "</think>" not in text:
        return ""
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    if "```" in text:
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL)
        if code_blocks:
            text = code_blocks[-1].strip()
    text = re.sub(r"^```[a-zA-Z]*\n", "", text, flags=re.DOTALL)
    text = re.sub(r"\n```$", "", text, flags=re.DOTALL)
    text = re.sub(r"^(final answer|answer)\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    bad_prefixes = (
        "here's",
        "the function",
        "###",
        "---",
    )
    lines = [ln.rstrip() for ln in text.splitlines()]
    if lines and lines[0].strip().lower().startswith(bad_prefixes):
        return ""
    return text


def generate_single(
    metanetwork,
    metalora,
    tokenizer,
    context: str,
    question: str,
    max_new_tokens: int,
    max_conversation_length: int,
    device: torch.device,
) -> tuple[str, str]:
    evidence_enc = tokenizer(
        [context],
        max_length=1024,
        truncation=True,
        return_tensors="pt",
        padding="max_length",
    )
    evidence_ids = evidence_enc["input_ids"].to(device, non_blocking=True)
    evidence_attention_mask = evidence_enc["attention_mask"].to(device, non_blocking=True)

    with torch.no_grad():
        lora_dict = metanetwork.generate_lora_dict(evidence_ids, evidence_attention_mask, metalora)
        messages = [
            {
                "role": "system",
                "content": (
                    "You complete Python function bodies. "
                    "Return ONLY the missing body lines with correct indentation. "
                    "No markdown, no explanation, no think mode."
                ),
            },
            {"role": "user", "content": question},
        ]
        try:
            input_enc = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                max_length=max_conversation_length,
                truncation=True,
                return_dict=True,
                padding="max_length",
                enable_thinking=False,
            )
        except TypeError:
            input_enc = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                max_length=max_conversation_length,
                truncation=True,
                return_dict=True,
                padding="max_length",
            )
        input_ids = input_enc["input_ids"].to(device)
        attention_mask = input_enc["attention_mask"].to(device)
        outputs = metanetwork.metamodel.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            ignore_mem_token=True,
            loradict=lora_dict,
        )
    new_tokens = outputs[0, input_ids.shape[1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return raw, extract_answer(raw)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    log = logging.getLogger("run_stage1a")

    if args.output_jsonl.exists() and not args.force:
        log.error("Output exists (use --force to overwrite): %s", args.output_jsonl)
        return 1
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_meta_json.parent.mkdir(parents=True, exist_ok=True)

    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model_path

    if not args.instances_jsonl.exists():
        raise FileNotFoundError(f"Missing instances JSONL: {args.instances_jsonl}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device=%s", device)
    log.info("Loading SHINE runtime from %s", args.vendor_shine_dir)
    runtime = load_shine_runtime(args.vendor_shine_dir)

    log.info(
        "Loading SHINE metanetwork + checkpoint. This can take several minutes on first run."
    )
    metanetwork, metalora, tokenizer = init_metanetwork(
        runtime=runtime,
        base_model_path=args.base_model_path,
        tokenizer_path=args.tokenizer_path,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
    )

    raw_rows = read_jsonl(args.instances_jsonl)
    rows = resolve_instances(raw_rows, repo_cache_dir=args.repo_cache_dir, log=log)
    rows = maybe_filter_instance_ids(rows, args.instance_ids_file, args.max_instances)
    log.info("Loaded %d instances for Stage 1a run.", len(rows))

    predictions: list[dict[str, Any]] = []
    coverage_values: list[float] = []

    progress = tqdm(rows, desc="Stage1a SHINE inference", unit="instance")
    for idx, row in enumerate(progress, start=1):
        iid = str(row.get("instance_id", f"row-{idx}"))
        t0 = time.perf_counter()
        try:
            slice_art = build_slice(
                row=row,
                tokenizer=tokenizer,
                max_tokens=args.context_max_tokens,
            )
            sig_doc = function_signature_plus_doc(
                str(row.get("function_source", "")),
                str(row.get("function_name", "")),
            )
            masked_function = str(row.get("masked_function", "")).strip()
            if masked_function:
                question = (
                    "Complete the missing function body.\n"
                    "Return ONLY body lines with indentation.\n\n"
                    f"Function signature + docstring:\n{sig_doc}\n\n"
                    f"Masked function:\n{masked_function}"
                )
            else:
                question = (
                    "Complete the missing function body.\n"
                    "Return ONLY body lines with indentation.\n\n"
                    f"Function signature + docstring:\n{sig_doc}"
                )
            raw_output, predicted_body = generate_single(
                metanetwork=metanetwork,
                metalora=metalora,
                tokenizer=tokenizer,
                context=slice_art.context_text,
                question=question,
                max_new_tokens=args.max_new_tokens,
                max_conversation_length=args.max_conversation_length,
                device=device,
            )
            elapsed = time.perf_counter() - t0
            coverage_values.append(slice_art.coverage_fraction)
            rec = {
                "instance_id": iid,
                "status": "ok",
                "repo": row.get("repo", ""),
                "file_path": row.get("file_path", ""),
                "function_name": row.get("function_name", ""),
                "start_line": row.get("start_line"),
                "end_line": row.get("end_line"),
                "masked_function": row.get("masked_function", ""),
                "ground_truth_body": row.get("ground_truth_body", ""),
                "full_file": row.get("full_file", ""),
                "gold_patch": row.get("gold_patch", ""),
                "docker_image": row.get("docker_image", ""),
                "test_cmd": row.get("test_cmd", ""),
                "question": question,
                "slice_context": slice_art.context_text,
                "slice_token_count": slice_art.token_count,
                "slice_coverage_fraction": slice_art.coverage_fraction,
                "covered_external_references": slice_art.covered_refs,
                "missing_external_references": slice_art.missing_refs,
                "raw_output": raw_output,
                "predicted_body": predicted_body,
                "generation_time_s": elapsed,
            }
            predictions.append(rec)
            if args.debug_every > 0 and idx % args.debug_every == 0:
                log.info(
                    "[%d/%d] %s | tokens=%d | coverage=%.3f | %.2fs",
                    idx,
                    len(rows),
                    iid,
                    rec["slice_token_count"],
                    rec["slice_coverage_fraction"],
                    elapsed,
                )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            log.exception("Failed %s after %.2fs", iid, elapsed)
            predictions.append(
                {
                    "instance_id": iid,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "generation_time_s": elapsed,
                }
            )

    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    median_cov = 0.0
    if coverage_values:
        sorted_cov = sorted(coverage_values)
        n = len(sorted_cov)
        mid = n // 2
        if n % 2 == 1:
            median_cov = sorted_cov[mid]
        else:
            median_cov = (sorted_cov[mid - 1] + sorted_cov[mid]) / 2.0

    summary = {
        "n_instances_requested": len(rows),
        "n_rows_written": len(predictions),
        "n_ok": sum(1 for r in predictions if r.get("status") == "ok"),
        "n_error": sum(1 for r in predictions if r.get("status") != "ok"),
        "slice_coverage_median": median_cov,
        "slice_coverage_target_met": bool(median_cov >= 0.70),
        "instances_jsonl": str(args.instances_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "config": {
            "max_instances": args.max_instances,
            "context_max_tokens": args.context_max_tokens,
            "max_new_tokens": args.max_new_tokens,
            "max_conversation_length": args.max_conversation_length,
            "base_model_path": str(args.base_model_path),
            "checkpoint_dir": str(args.checkpoint_dir),
            "vendor_shine_dir": str(args.vendor_shine_dir),
            "seed": args.seed,
            "device": str(device),
        },
    }
    args.output_meta_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info("Saved predictions to %s", args.output_jsonl)
    log.info("Saved run metadata to %s", args.output_meta_json)
    log.info(
        "Slice coverage median = %.3f (target >= 0.70 => %s)",
        median_cov,
        "PASS" if summary["slice_coverage_target_met"] else "FAIL",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
