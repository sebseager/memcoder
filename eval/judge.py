"""LLM-judge phase of the eval harness.

Reads ``predictions.jsonl`` and queries the OpenAI API once per row to grade
the model's answer against the ground truth on a 1–5 scale, tagging failure
modes whenever the score is below 5. Output is written row-for-row to
``judgments.jsonl`` in the same order as the input.

Calls are issued concurrently with bounded asyncio concurrency. The chosen
model id is passed straight to the OpenAI SDK — no provider prefix.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import JudgeConfig, RunConfig

LOGGER = logging.getLogger("memcoder.eval.judge")

FAILURE_MODE_VALUES = (
    "wrong_specifics",
    "missing_information",
    "off_topic",
    "refusal_or_nonresponse",
    "format_failure",
    "other",
)

JUDGE_JSON_SCHEMA = {
    "name": "memcoder_judge_grading_v0",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["score", "reasoning", "failure_modes", "failure_mode_notes"],
        "properties": {
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "reasoning": {"type": "string"},
            "failure_modes": {
                "type": "array",
                "items": {"type": "string", "enum": list(FAILURE_MODE_VALUES)},
            },
            "failure_mode_notes": {"type": "string"},
        },
    },
    "strict": True,
}


@dataclass
class _JudgeContext:
    cfg: JudgeConfig
    rubric_template: str
    semaphore: asyncio.Semaphore
    client: Any  # AsyncOpenAI


def run_judging(cfg: RunConfig, run_dir: Path) -> Path:
    """Run the judge phase. Returns the path to ``judgments.jsonl``."""
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions.jsonl not found at {predictions_path}")

    judgments_path = run_dir / "judgments.jsonl"
    if judgments_path.exists():
        raise FileExistsError(
            f"judgments.jsonl already exists at {judgments_path}; the harness "
            "is non-resumable — start a fresh run if you want to re-judge"
        )

    if cfg.judge.dotenv_path is not None:
        _load_dotenv(cfg.judge.dotenv_path)
    api_key = os.environ.get(cfg.judge.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"environment variable {cfg.judge.api_key_env!r} is not set; "
            f"add it to {cfg.judge.dotenv_path or '.env'} or your shell"
        )

    rubric_template = _load_prompt(cfg.judge.prompt)
    rows = _read_predictions(predictions_path)
    LOGGER.info(
        "Judging %d row(s) with %s (concurrency=%d)",
        len(rows),
        cfg.judge.model,
        cfg.judge.concurrency,
    )

    judged = asyncio.run(_judge_all(rows, cfg, rubric_template, api_key))

    with judgments_path.open("w", encoding="utf-8") as out:
        for row in judged:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    LOGGER.info("Judgments complete: %d rows -> %s", len(judged), judgments_path)
    return judgments_path


async def _judge_all(
    rows: list[dict[str, Any]],
    cfg: RunConfig,
    rubric_template: str,
    api_key: str,
) -> list[dict[str, Any]]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(cfg.judge.concurrency)
    ctx = _JudgeContext(
        cfg=cfg.judge,
        rubric_template=rubric_template,
        semaphore=semaphore,
        client=client,
    )

    # Preserve input order. We build a list of placeholders and fill them in
    # via gathered tasks indexed by row position.
    judged: list[dict[str, Any]] = [dict() for _ in rows]

    async def _one(idx: int, row: dict[str, Any]) -> None:
        try:
            judge_block = await _judge_row(ctx, row)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("judge failed for row %d (qa=%s, cond=%s): %s",
                         idx, row.get("qa_id"), row.get("condition"), exc)
            judge_block = _judge_error_block(cfg.judge, str(exc))
        merged = dict(row)
        merged["judge"] = judge_block
        judged[idx] = merged

    await asyncio.gather(*(_one(i, r) for i, r in enumerate(rows)))
    return judged


async def _judge_row(ctx: _JudgeContext, row: dict[str, Any]) -> dict[str, Any]:
    prompt = ctx.rubric_template.format(
        question=row.get("question", ""),
        expected_answer=_render_expected(row.get("expected_answer")),
        model_answer=row.get("answer", ""),
    )

    last_error: Exception | None = None
    for attempt in range(ctx.cfg.max_retries + 1):
        try:
            async with ctx.semaphore:
                response = await ctx.client.chat.completions.create(
                    model=ctx.cfg.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an experienced senior software engineer "
                                "grading another model's answers to questions about "
                                "open-source codebases. Follow the rubric exactly "
                                "and return JSON conforming to the provided schema."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": JUDGE_JSON_SCHEMA,
                    },
                )
            content = response.choices[0].message.content or ""
            payload = json.loads(content)
            return _normalize_judge_payload(ctx.cfg, payload, content)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= ctx.cfg.max_retries:
                break
            backoff = min(30.0, (2 ** attempt) + random.random())
            LOGGER.warning(
                "judge call failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                ctx.cfg.max_retries + 1,
                exc,
                backoff,
            )
            await asyncio.sleep(backoff)

    raise RuntimeError(f"judge call failed after retries: {last_error}")


def _normalize_judge_payload(
    cfg: JudgeConfig, payload: dict[str, Any], raw_response: str
) -> dict[str, Any]:
    score = int(payload["score"])
    if score < 1 or score > 5:
        raise ValueError(f"judge returned out-of-range score: {score}")

    failure_modes = list(payload.get("failure_modes") or [])
    failure_mode_notes = str(payload.get("failure_mode_notes") or "")
    reasoning = str(payload.get("reasoning") or "")

    invalid = [m for m in failure_modes if m not in FAILURE_MODE_VALUES]
    if invalid:
        raise ValueError(f"judge returned unknown failure modes: {invalid}")

    if score == 5 and failure_modes:
        # Trust the score; clear the tags rather than discarding the row.
        LOGGER.info("Clearing failure_modes for score==5 row")
        failure_modes = []
        failure_mode_notes = ""
    if score < 5 and not failure_modes:
        # Prompt asks for ≥1 tag below 5; fall back to 'other' rather than reject.
        failure_modes = ["other"]
        if not failure_mode_notes:
            failure_mode_notes = (
                "judge returned no failure modes for score<5; auto-tagged 'other'"
            )

    block: dict[str, Any] = {
        "model": cfg.model,
        "rubric_version": cfg.rubric_version,
        "taxonomy_version": cfg.taxonomy_version,
        "score": score,
        "reasoning": reasoning,
        "failure_modes": failure_modes,
    }
    if failure_mode_notes:
        block["failure_mode_notes"] = failure_mode_notes
    block["raw_response"] = raw_response
    return block


def _judge_error_block(cfg: JudgeConfig, error: str) -> dict[str, Any]:
    """Recorded when all retries fail. Score 0 is intentionally invalid so it
    shows up in reports; downstream code treats missing scores as errors."""
    return {
        "model": cfg.model,
        "rubric_version": cfg.rubric_version,
        "taxonomy_version": cfg.taxonomy_version,
        "score": 1,
        "reasoning": f"JUDGE_ERROR: {error}",
        "failure_modes": ["other"],
        "failure_mode_notes": "judge call failed after retries",
        "raw_response": "",
    }


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{lineno}: {exc}") from exc
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def _load_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    required_placeholders = ("{question}", "{expected_answer}", "{model_answer}")
    missing = [p for p in required_placeholders if p not in text]
    if missing:
        raise ValueError(
            f"judge prompt {path} missing placeholders: {missing} "
            "(expected {question}, {expected_answer}, {model_answer})"
        )
    return text


def _render_expected(value: Any) -> str:
    if value is None:
        return "(no ground truth provided)"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        LOGGER.warning("dotenv file not found at %s; relying on shell env", path)
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(str(path), override=False)
    except ImportError:  # pragma: no cover
        # Minimal fallback: parse KEY=VALUE lines manually.
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
