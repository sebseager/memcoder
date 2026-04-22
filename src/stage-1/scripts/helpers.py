from __future__ import annotations

import ast
import hashlib
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from config import (
    ENABLE_THINKING,
    INSTANCES_JSONL,
    LAYER_TARGETING_DECISION,
    LAYER_TARGETING_JUSTIFICATION,
    LAYER_TARGETING_REFERENCE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_TARGET_MODULES,
    MODEL_ID,
    ORACLE_MIN_CHUNK_TOKENS,
    SEED,
    TRUNCATION_BUDGET_TOKENS,
)

# Torch / transformers / peft are imported lazily inside the functions that
# actually need them. This keeps the pure-Python utilities in this module
# (artifact hydration, AST-based masked-function extraction, supervised
# record construction, etc.) importable in environments that haven't yet
# installed the heavy ML stack.


@dataclass
class FileRecord:
    file_key: str
    repo: str
    file_path: str
    full_file: str


@dataclass
class FunctionExample:
    function_name: str
    masked_function: str
    ground_truth_body: str
    context_text: str
    source: str
    start_line: int | None = None
    end_line: int | None = None


def set_seed(seed: int = SEED) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_sha256(payload: dict | list) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def build_run_config(
    *,
    seed: int,
    model_id: str,
    truncation_budget_tokens: int,
    oracle_chunk_size: int,
    behavioral_probes: int,
    behavioral_epochs: int,
    behavioral_lr_mult: float,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    notes: dict | None = None,
) -> dict:
    config = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "model_id": model_id,
        "truncation_budget_tokens": truncation_budget_tokens,
        "oracle_chunk_size": oracle_chunk_size,
        "behavioral_probes": behavioral_probes,
        "behavioral_epochs": behavioral_epochs,
        "behavioral_lr_mult": behavioral_lr_mult,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "layer_targets": LORA_TARGET_MODULES,
        "layer_targeting": {
            "decision": LAYER_TARGETING_DECISION,
            "justification": LAYER_TARGETING_JUSTIFICATION,
            "reference": LAYER_TARGETING_REFERENCE,
        },
    }
    if notes:
        config["notes"] = notes
    return config


def _extract_masked_function_text(
    function_source_text: str,
    function_name: str,
) -> str:
    """Return the signature lines followed by a single ``pass`` body.

    Stage 1's internal contract (see ``_extract_function_example_from_node``)
    treats ``masked_function`` as the declaration lines only -- decorators plus
    the (possibly multi-line) ``def`` through the colon -- with a single
    ``pass`` body at the body's indent. Any docstring belongs to
    ``ground_truth_body``.

    Stage 0's ``masked_file_artifact`` preserves the docstring, so AST-parsing
    that file would incorrectly include the docstring here and lead to a
    duplicated docstring when we rebuild the function body during
    evaluation. Parse ``function_source`` instead (it's the original
    unmasked function text, starting at line 1 of ``function_source``).
    """

    import textwrap

    # function_source may be indented (e.g. a method inside a class) because it
    # is extracted verbatim from the original file. Dedent only for parsing; use
    # the original text when slicing lines so indentation is preserved.
    dedented = textwrap.dedent(function_source_text)
    try:
        tree = ast.parse(dedented)
    except SyntaxError as exc:
        raise ValueError(
            f"function_source for {function_name!r} did not parse: {exc}"
        ) from exc

    target: ast.AST | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                target = node
                break
    if target is None:
        # Fall back to the first function in the source; function_source is
        # expected to contain a single function definition.
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                target = node
                break
    if target is None:
        raise ValueError(
            f"No function definition in function_source for {function_name!r}"
        )

    lines = function_source_text.splitlines()
    body = getattr(target, "body", None)
    if not body:
        raise ValueError(
            f"Function {function_name!r} has no body in function_source"
        )
    body_start_line = int(getattr(body[0], "lineno"))
    start_line = int(getattr(target, "lineno"))
    decorator_lines = [
        int(getattr(d, "lineno", start_line))
        for d in getattr(target, "decorator_list", [])
    ]
    if decorator_lines:
        start_line = min([start_line] + decorator_lines)

    signature_lines = lines[start_line - 1 : body_start_line - 1]
    if not signature_lines:
        raise ValueError(
            f"Could not extract signature for {function_name!r}"
        )

    first_body_line = lines[body_start_line - 1] if body_start_line - 1 < len(lines) else ""
    indent_match = re.match(r"^\s*", first_body_line)
    body_indent = indent_match.group(0) if indent_match is not None else "    "
    if not body_indent:
        def_indent_match = re.match(r"^\s*", signature_lines[0])
        body_indent = (def_indent_match.group(0) if def_indent_match else "") + "    "

    return "\n".join(signature_lines).rstrip() + "\n" + body_indent + "pass"


def _read_artifact_text(path_str: str | None) -> str:
    if not path_str:
        return ""
    path = Path(path_str)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def hydrate_instance_row(row: dict) -> dict:
    """Inject full_file / masked_file / masked_function / ground_truth_body.

    Stage 0 now emits artifacts on disk (`full_file_artifact`, `masked_file_artifact`,
    `ground_truth_artifact`, `function_source_artifact`) rather than inlining file
    text in the JSONL row. Older Stage 1 code assumes the inlined fields, so we
    hydrate the row in-place when artifacts are present, preserving any already
    inlined fields if they exist.
    """

    full_file = row.get("full_file")
    if not full_file:
        full_file = _read_artifact_text(row.get("full_file_artifact"))
        if full_file:
            row["full_file"] = full_file

    masked_file = row.get("masked_file")
    if not masked_file:
        masked_file = _read_artifact_text(row.get("masked_file_artifact"))
        if masked_file:
            row["masked_file"] = masked_file

    ground_truth = row.get("ground_truth_body")
    if not ground_truth:
        ground_truth = _read_artifact_text(row.get("ground_truth_artifact"))
        if ground_truth:
            row["ground_truth_body"] = ground_truth

    function_source = row.get("function_source")
    if not function_source:
        function_source = _read_artifact_text(row.get("function_source_artifact"))
        if function_source:
            row["function_source"] = function_source

    if not row.get("masked_function") and function_source and row.get("function_name"):
        try:
            row["masked_function"] = _extract_masked_function_text(
                function_source_text=function_source,
                function_name=str(row["function_name"]),
            )
        except ValueError:
            row["masked_function"] = ""

    return row


def load_instances(instances_path: Path = INSTANCES_JSONL) -> list[dict]:
    rows = []
    with instances_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            hydrate_instance_row(row)
            rows.append(row)
    return rows


def make_file_key(repo: str, file_path: str) -> str:
    raw = f"{repo}__{file_path}"
    # Keep keys filesystem-safe and deterministic.
    key = re.sub(r"[^A-Za-z0-9_.-]", "_", raw)
    return key


def build_file_records(instances: list[dict]) -> list[FileRecord]:
    by_key: dict[str, FileRecord] = {}
    for row in instances:
        repo = row["repo"]
        file_path = row["file_path"]
        key = make_file_key(repo, file_path)
        if key not in by_key:
            by_key[key] = FileRecord(
                file_key=key,
                repo=repo,
                file_path=file_path,
                full_file=row["full_file"],
            )
    return sorted(by_key.values(), key=lambda x: x.file_key)


def select_instances(instances: list[dict], max_instances: int | None) -> list[dict]:
    if max_instances is None:
        return instances
    return instances[:max_instances]


def select_file_records(
    file_records: list[FileRecord], max_files: int | None
) -> list[FileRecord]:
    if max_files is None:
        return file_records
    return file_records[:max_files]


def truncate_to_budget(text: str, tokenizer, budget_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= budget_tokens:
        return text

    marker = "\n# ... [truncated to token budget] ...\n"
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    keep_tokens = max(0, budget_tokens - len(marker_ids))

    clipped = tokenizer.decode(ids[:keep_tokens], skip_special_tokens=True)
    cut = clipped.rfind("\n")
    if cut > 0:
        clipped = clipped[: cut + 1]

    merged = clipped + marker
    merged_ids = tokenizer.encode(merged, add_special_tokens=False)
    if len(merged_ids) > budget_tokens:
        merged = tokenizer.decode(merged_ids[:budget_tokens], skip_special_tokens=True)
    return merged


def make_lora_config():
    from peft import LoraConfig

    return LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
    )


def load_model_and_tokenizer(
    model_id: str = MODEL_ID,
    use_4bit: bool = True,
    seed: int = SEED,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_cfg,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    # Avoid warning spam under greedy decoding by restoring neutral defaults.
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0
        model.generation_config.top_k = 50

    return model, tokenizer


def prepare_model_for_oracle_training(model):
    from peft import prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    return model


def cycle_lora_adapter(peft_model, adapter_name: str = "default"):
    peft_model.delete_adapter(adapter_name)
    peft_model.add_adapter(adapter_name, make_lora_config())
    peft_model.set_adapter(adapter_name)
    return peft_model


def make_chat_prompt(
    masked_function: str, context_text: str, function_name: str
) -> tuple[str, str]:
    system_prompt = (
        "You complete Python function bodies. "
        "Return only the function body lines with correct indentation. "
        "Do not include markdown fences."
    )
    user_prompt = (
        f"Target function name: {function_name}\n\n"
        "Masked function:\n"
        f"{masked_function}\n\n"
        "Relevant file context:\n"
        f"{context_text}\n\n"
        "Output only the missing function body for the masked function."
    )
    return system_prompt, user_prompt


def generate_text(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    do_sample = temperature > 0
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(**inputs, **kwargs)

    gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def normalize_body_prediction(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)

    # If the model returns an entire function, keep only body lines.
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("def "):
        lines = lines[1:]

    # Drop obvious role/preamble artifacts.
    while lines and lines[0].strip().lower().startswith(("assistant", "here is")):
        lines = lines[1:]

    # Ensure body indentation exists.
    norm = []
    for line in lines:
        if not line.strip():
            norm.append("")
            continue
        if line.startswith("    "):
            norm.append(line.rstrip())
        else:
            norm.append(f"    {line.rstrip()}")
    return "\n".join(norm).rstrip()


def _line_indent(line: str) -> str:
    match = re.match(r"^\s*", line)
    return "" if match is None else match.group(0)


def _replace_line_span(
    lines: list[str],
    start_line: int,
    end_line: int,
    replacement_lines: list[str],
) -> list[str]:
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    return lines[:start_idx] + replacement_lines + lines[end_idx:]


def masked_file_context_from_span(
    full_file: str,
    start_line: int,
    end_line: int,
    masked_function: str,
    tokenizer,
    trunc_budget_tokens: int = TRUNCATION_BUDGET_TOKENS,
) -> str:
    lines = full_file.splitlines()
    masked_lines = masked_function.splitlines()
    replaced = _replace_line_span(lines, start_line, end_line, masked_lines)
    masked_file_text = "\n".join(replaced)
    return truncate_to_budget(masked_file_text, tokenizer, trunc_budget_tokens)


def _collect_function_nodes(tree: ast.AST) -> list[ast.AST]:
    nodes: list[ast.AST] = []

    class _Collector(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            nodes.append(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            nodes.append(node)
            self.generic_visit(node)

    _Collector().visit(tree)
    return sorted(nodes, key=lambda n: getattr(n, "lineno", 0))


def _extract_function_example_from_node(
    file_text: str,
    node: ast.AST,
    tokenizer,
    trunc_budget_tokens: int,
) -> FunctionExample | None:
    if not hasattr(node, "body") or not getattr(node, "body"):
        return None
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return None

    body = getattr(node, "body")
    first_body_node = body[0]
    if not hasattr(first_body_node, "lineno"):
        return None

    name = getattr(node, "name", None)
    if not isinstance(name, str) or not name:
        return None

    lines = file_text.splitlines()
    decorator_lines = [
        getattr(d, "lineno", getattr(node, "lineno"))
        for d in getattr(node, "decorator_list", [])
    ]
    start_line = min([getattr(node, "lineno")] + decorator_lines)
    end_line = int(getattr(node, "end_lineno"))
    body_start_line = int(getattr(first_body_node, "lineno"))

    if start_line < 1 or end_line > len(lines) or body_start_line <= start_line:
        return None

    signature_lines = lines[start_line - 1 : body_start_line - 1]
    body_lines = lines[body_start_line - 1 : end_line]
    if not signature_lines or not body_lines:
        return None

    body_indent = _line_indent(body_lines[0])
    masked_function = "\n".join(signature_lines).rstrip() + "\n" + body_indent + "pass"
    ground_truth_body = "\n".join(body_lines).rstrip()
    context_text = masked_file_context_from_span(
        full_file=file_text,
        start_line=start_line,
        end_line=end_line,
        masked_function=masked_function,
        tokenizer=tokenizer,
        trunc_budget_tokens=trunc_budget_tokens,
    )

    return FunctionExample(
        function_name=name,
        masked_function=masked_function,
        ground_truth_body=ground_truth_body,
        context_text=context_text,
        source="ast",
        start_line=start_line,
        end_line=end_line,
    )


def build_function_examples_from_file(
    file_text: str,
    tokenizer,
    trunc_budget_tokens: int = TRUNCATION_BUDGET_TOKENS,
) -> list[FunctionExample]:
    try:
        tree = ast.parse(file_text)
    except SyntaxError:
        return []

    examples: list[FunctionExample] = []
    for node in _collect_function_nodes(tree):
        ex = _extract_function_example_from_node(
            file_text=file_text,
            node=node,
            tokenizer=tokenizer,
            trunc_budget_tokens=trunc_budget_tokens,
        )
        if ex is None:
            continue
        if ex.ground_truth_body.strip() == "":
            continue
        examples.append(ex)
    return examples


def build_function_examples_from_instances(
    instances_for_file: list[dict],
    tokenizer,
    trunc_budget_tokens: int = TRUNCATION_BUDGET_TOKENS,
) -> list[FunctionExample]:
    rows = sorted(instances_for_file, key=lambda r: r.get("instance_id", ""))
    out: list[FunctionExample] = []
    for row in rows:
        context_text = masked_file_context_from_span(
            full_file=row["full_file"],
            start_line=int(row["start_line"]),
            end_line=int(row["end_line"]),
            masked_function=row["masked_function"],
            tokenizer=tokenizer,
            trunc_budget_tokens=trunc_budget_tokens,
        )
        out.append(
            FunctionExample(
                function_name=row["function_name"],
                masked_function=row["masked_function"],
                ground_truth_body=row["ground_truth_body"],
                context_text=context_text,
                source="instance_fallback",
                start_line=int(row.get("start_line", 0))
                if row.get("start_line")
                else None,
                end_line=int(row.get("end_line", 0)) if row.get("end_line") else None,
            )
        )
    return out


def build_behavioral_probe_manifest(
    examples: list[FunctionExample],
    n_probes: int,
) -> dict:
    selected = examples[: max(0, n_probes)]
    probes = []
    for idx, ex in enumerate(selected):
        probes.append(
            {
                "probe_index": idx,
                "function_name": ex.function_name,
                "source": ex.source,
                "start_line": ex.start_line,
                "end_line": ex.end_line,
                "masked_function_sha256": hashlib.sha256(
                    ex.masked_function.encode("utf-8")
                ).hexdigest(),
                "ground_truth_body_sha256": hashlib.sha256(
                    ex.ground_truth_body.encode("utf-8")
                ).hexdigest(),
            }
        )

    manifest = {
        "generator": "ast_order_first_n",
        "n_requested": int(n_probes),
        "n_selected": len(probes),
        "probes": probes,
    }
    manifest["manifest_sha256"] = json_sha256(manifest)
    return manifest


def _apply_chat_template_for_training(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )


def build_supervised_records_from_examples(
    examples: list[FunctionExample],
    tokenizer,
    max_sequence_tokens: int | None,
) -> tuple[list[dict], dict]:
    records: list[dict] = []
    stats = {
        "n_examples": len(examples),
        "n_records": 0,
        "n_truncated": 0,
        "mean_prompt_tokens": 0.0,
        "mean_target_tokens": 0.0,
    }

    prompt_lens = []
    target_lens = []

    for ex in examples:
        system_prompt, user_prompt = make_chat_prompt(
            masked_function=ex.masked_function,
            context_text=ex.context_text,
            function_name=ex.function_name,
        )
        prompt_text = _apply_chat_template_for_training(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        target_ids = tokenizer.encode(ex.ground_truth_body, add_special_tokens=False)
        if not target_ids:
            continue

        if max_sequence_tokens is not None:
            total_len = len(prompt_ids) + len(target_ids)
            if total_len > max_sequence_tokens:
                overflow = total_len - max_sequence_tokens
                if overflow < len(prompt_ids):
                    prompt_ids = prompt_ids[overflow:]
                else:
                    trim_target = overflow - len(prompt_ids)
                    prompt_ids = []
                    keep_target = max(1, len(target_ids) - trim_target)
                    target_ids = target_ids[:keep_target]
                stats["n_truncated"] += 1

        input_ids = prompt_ids + target_ids
        labels = ([-100] * len(prompt_ids)) + target_ids
        records.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
        )
        prompt_lens.append(len(prompt_ids))
        target_lens.append(len(target_ids))

    stats["n_records"] = len(records)
    if prompt_lens:
        stats["mean_prompt_tokens"] = float(np.mean(prompt_lens))
    if target_lens:
        stats["mean_target_tokens"] = float(np.mean(target_lens))
    return records, stats


def inspect_first_supervised_example(
    examples: list[FunctionExample],
    tokenizer,
    max_sequence_tokens: int | None,
) -> dict | None:
    if not examples:
        return None

    records, _ = build_supervised_records_from_examples(
        examples=[examples[0]],
        tokenizer=tokenizer,
        max_sequence_tokens=max_sequence_tokens,
    )
    if not records:
        return None

    rec = records[0]
    labels = rec["labels"]
    supervised_tokens = sum(1 for x in labels if x != -100)
    masked_tokens = sum(1 for x in labels if x == -100)
    return {
        "function_name": examples[0].function_name,
        "source": examples[0].source,
        "masked_function": examples[0].masked_function,
        "target_body": examples[0].ground_truth_body,
        "decoded_input": tokenizer.decode(rec["input_ids"], skip_special_tokens=True),
        "supervised_token_count": supervised_tokens,
        "masked_prompt_token_count": masked_tokens,
        "total_input_tokens": len(rec["input_ids"]),
    }


def make_chunk_dataset_records(
    text: str,
    tokenizer,
    chunk_size: int,
) -> list[dict]:
    pad_id = tokenizer.pad_token_id
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start : start + chunk_size]
        if len(chunk) < ORACLE_MIN_CHUNK_TOKENS:
            continue
        pad_n = chunk_size - len(chunk)
        chunks.append(
            {
                "input_ids": chunk + [pad_id] * pad_n,
                "attention_mask": [1] * len(chunk) + [0] * pad_n,
                "labels": chunk + [-100] * pad_n,
            }
        )
    return chunks
