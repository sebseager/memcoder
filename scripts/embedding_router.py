from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed questions and LoRA descriptions, then route to top-k LoRAs.")
    p.add_argument("--lora-store", type=Path, required=True)
    p.add_argument("--question", type=str, default=None)
    p.add_argument("--qa-pairs", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def load_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def flatten_loras(store: Any, store_path: Path | None = None) -> list[dict[str, Any]]:
    """
    Supports:
    1. [{"lora_id": ...}]
    2. {"loras": [{"lora_id": ...}]}
    3. {"loras": {"overview_purpose_1": {...}}}
    4. {"documents": {"overview_purpose_1": {...}}}  # ledger.json format
    """
    base_dir = store_path.parent if store_path is not None else Path(".")

    if isinstance(store, list):
        return store

    if isinstance(store, dict):
        if isinstance(store.get("documents"), dict):
            out = []
            for doc_id, meta in store["documents"].items():
                row = dict(meta)
                row.setdefault("document_id", doc_id)
                row.setdefault("lora_id", doc_id)

                files = row.get("files", {}) or {}

                if "doc" in files and files["doc"]:
                    row["source_doc"] = str((base_dir / files["doc"]).resolve())
                if "qa" in files and files["qa"]:
                    row["qa_file"] = str((base_dir / files["qa"]).resolve())
                if "lora" in files and files["lora"]:
                    row["lora_path"] = str((base_dir / files["lora"]).resolve())
                if "doc_embedding" in files and files["doc_embedding"]:
                    row["embedding_path"] = str((base_dir / files["doc_embedding"]).resolve())

                out.append(row)
            return out

        loras = store.get("loras", store.get("entries", []))

        if isinstance(loras, list):
            return loras

        if isinstance(loras, dict):
            out = []
            for lora_id, meta in loras.items():
                row = dict(meta)
                row.setdefault("lora_id", lora_id)

                files = row.get("files", {}) or {}

                if "doc" in files and files["doc"]:
                    row["source_doc"] = str((base_dir / files["doc"]).resolve())
                if "qa" in files and files["qa"]:
                    row["qa_file"] = str((base_dir / files["qa"]).resolve())
                if "lora" in files and files["lora"]:
                    row["lora_path"] = str((base_dir / files["lora"]).resolve())
                if "embedding" in files and files["embedding"]:
                    row["embedding_path"] = str((base_dir / files["embedding"]).resolve())

                out.append(row)
            return out

    raise ValueError("Unsupported lora_store/ledger format")


def read_source_doc_text(source_doc: str | None) -> str:
    if not source_doc:
        return ""

    path = Path(source_doc)
    if not path.exists():
        return ""

    try:
        obj = load_json_or_jsonl(path)
    except Exception:
        return path.read_text(encoding="utf-8", errors="ignore")

    if isinstance(obj, dict):
        for key in ["document", "content", "text", "body", "summary"]:
            if key in obj and isinstance(obj[key], str):
                return obj[key]
        return json.dumps(obj, ensure_ascii=False)

    return json.dumps(obj, ensure_ascii=False)


def make_lora_routing_text(lora: dict[str, Any]) -> str:
    title = lora.get("title") or lora.get("topic") or lora.get("lora_id", "")
    description = lora.get("description") or lora.get("summary") or lora.get("intended_knowledge", "")

    example_questions = lora.get("example_questions", [])
    if isinstance(example_questions, list):
        example_questions_text = "\n".join(f"- {q}" for q in example_questions)
    else:
        example_questions_text = str(example_questions)

    keywords = lora.get("keywords", [])
    if isinstance(keywords, list):
        keywords_text = ", ".join(str(k) for k in keywords)
    else:
        keywords_text = str(keywords)

    source_doc = lora.get("source_doc")
    source_text = read_source_doc_text(source_doc)

    return f"""LoRA ID: {lora.get("lora_id", "")}
Title: {title}
Description: {description}
Keywords: {keywords_text}
Example questions:
{example_questions_text}

Source document contents:
{source_text}
""".strip()


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def embed_texts(texts: list[str], tokenizer, model, device: str, max_length: int | None = None) -> torch.Tensor:
    if max_length is None:
        tok_max = getattr(tokenizer, "model_max_length", 512) or 512
        model_max = getattr(getattr(model, "config", None), "max_position_embeddings", tok_max) or tok_max

        # Some tokenizers report an absurd sentinel value for "unlimited".
        if tok_max > 100_000:
            tok_max = model_max

        max_length = min(tok_max, model_max, 8192)

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    output = model(**encoded)
    emb = mean_pool(output.last_hidden_state, encoded["attention_mask"])
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu()


def cosine_scores(question_emb: torch.Tensor, lora_embs: torch.Tensor) -> list[float]:
    scores = (lora_embs @ question_emb.squeeze(0)).tolist()
    return [float(s) for s in scores]


def route_question(
    question: str,
    loras: list[dict[str, Any]],
    tokenizer,
    model,
    device: str,
    top_k: int,
) -> dict[str, Any]:
    routing_texts = [make_lora_routing_text(lora) for lora in loras]

    q_emb = embed_texts([question], tokenizer, model, device)
    l_embs = embed_texts(routing_texts, tokenizer, model, device)

    scores = cosine_scores(q_emb, l_embs)
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
        "top_k": top_k,
        "selected_lora_ids": [r["lora_id"] for r in selected],
        "ranked_loras": ranked,
    }


def iter_questions(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.question:
        return [{"question": args.question, "qa_id": None, "gold_lora_id": None}]

    if args.qa_pairs:
        rows = load_json_or_jsonl(args.qa_pairs)
        if isinstance(rows, dict):
            rows = (
                rows.get("questions")
                or rows.get("qas")
                or rows.get("qa_pairs")
                or rows.get("items")
                or []
            )

        out = []
        for row in rows:
            out.append(
                {
                    "question": row.get("question", ""),
                    "qa_id": row.get("qa_id") or row.get("question_id"),
                    "gold_lora_id": row.get("gold_lora_id") or row.get("source_lora_id") or row.get("document_id"),
                }
            )
        return out

    raise ValueError("Provide either --question or --qa-pairs")


def main() -> int:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    store = load_json_or_jsonl(args.lora_store)
    loras = flatten_loras(store, args.lora_store)

    if not loras:
        raise ValueError(f"No LoRAs found in {args.lora_store}")

    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.embedding_model,
        trust_remote_code=True,
        dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
    ).to(args.device)
    model.eval()

    results = []
    for item in iter_questions(args):
        result = route_question(
            question=item["question"],
            loras=loras,
            tokenizer=tokenizer,
            model=model,
            device=args.device,
            top_k=args.top_k,
        )
        result["qa_id"] = item.get("qa_id")
        result["gold_lora_id"] = item.get("gold_lora_id")
        if result["gold_lora_id"]:
            result["top1_correct"] = result["ranked_loras"][0]["lora_id"] == result["gold_lora_id"]
            result["topk_correct"] = result["gold_lora_id"] in result["selected_lora_ids"]
        results.append(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(results[0] if len(results) == 1 else {"n": len(results), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())