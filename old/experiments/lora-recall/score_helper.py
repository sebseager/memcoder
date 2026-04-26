"""Scoring helper for LoRA recall experiments.

Provides a smarter answer-matching system than naive exact match.
Handles common mismatches:
  - Number words ↔ digits  ("7" vs "Seven")
  - Surrounding quotes/backticks  (`FlorpException` vs FlorpException)
  - Embedded answers in verbose responses ("The answer is 7." vs "7")
  - Punctuation / whitespace differences
  - Ordinals ("1st" vs "first")

Usage:
  from score_helper import score, is_correct

  result = score("The answer is seven.", "7")
  # result.correct == True, result.method == "contains_normalized"

  if is_correct("FlorpException", "FlorpException"):
      ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Number normalization tables
# ---------------------------------------------------------------------------
_DIGIT_TO_WORD = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "11": "eleven",
    "12": "twelve",
    "13": "thirteen",
    "14": "fourteen",
    "15": "fifteen",
    "16": "sixteen",
    "17": "seventeen",
    "18": "eighteen",
    "19": "nineteen",
    "20": "twenty",
}

_WORD_TO_DIGIT: dict[str, str] = {v: k for k, v in _DIGIT_TO_WORD.items()}

_ORDINAL_TO_CARDINAL = {
    "first": "one",
    "second": "two",
    "third": "three",
    "fourth": "four",
    "fifth": "five",
    "sixth": "six",
    "seventh": "seven",
    "eighth": "eight",
    "ninth": "nine",
    "tenth": "ten",
    "eleventh": "eleven",
    "twelfth": "twelve",
}

_ORDINAL_SUFFIX_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def _strip_formatting(text: str) -> str:
    """Remove backticks, quotes, and markdown bold/italic markers."""
    text = re.sub(r"[`\"']", "", text)
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)  # **bold** / *italic*
    return text


def _normalize_numbers(text: str) -> str:
    """Convert digit strings → words and ordinal suffixes → cardinal words."""
    # "1st" / "2nd" / "3rd" / "4th" → digit only
    text = _ORDINAL_SUFFIX_RE.sub(r"\1", text)
    # Digit sequences → words (where we have a mapping)
    text = re.sub(
        r"\b\d+\b",
        lambda m: _DIGIT_TO_WORD.get(m.group(0), m.group(0)),
        text,
    )
    # Word ordinals → cardinal words
    text = re.sub(
        r"\b(" + "|".join(_ORDINAL_TO_CARDINAL) + r")\b",
        lambda m: _ORDINAL_TO_CARDINAL[m.group(0)],
        text,
    )
    return text


def normalize(text: str) -> str:
    """Lowercase, strip formatting, normalize numbers and whitespace."""
    text = text.lower().strip()
    text = _strip_formatting(text)
    text = _normalize_numbers(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_words_only(text: str) -> str:
    """Normalize and also convert word-form numbers to digits for comparison."""
    text = normalize(text)
    # Also map words → digit strings so "seven" and "7" both become "seven"
    # (since _normalize_numbers already maps digits→words, this is a no-op
    #  in the normal path, but handles gold="seven" vs pred="7")
    return text


# ---------------------------------------------------------------------------
# Scoring result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoreResult:
    """Result of comparing a predicted answer against a gold answer."""

    correct: bool
    method: str  # which check matched: "exact", "contains", "none"
    gold_normalized: str
    predicted_normalized: str


def score(predicted: str, gold: str) -> ScoreResult:
    """Score a predicted answer against a gold answer.

    Checks in order:
      1. **Exact normalized match**: after normalizing both, they're equal.
      2. **Contains normalized**: the normalized gold appears as a substring
         of the normalized prediction.  Handles verbose model responses
         that wrap the answer in a sentence.

    Returns a ``ScoreResult`` with ``.correct`` and ``.method``.
    """
    pred_norm = normalize(predicted)
    gold_norm = normalize(gold)

    # Also build a variant where word-numbers are mapped back to digits
    # so we can match "seven" in prediction against "7" in gold and vice-versa
    gold_word = _normalize_numbers(gold.lower().strip())
    gold_word = _strip_formatting(gold_word)
    gold_word = re.sub(r"\s+", " ", gold_word).strip()

    # 1. Exact match after normalization
    if pred_norm == gold_norm:
        return ScoreResult(True, "exact", gold_norm, pred_norm)

    # 2. Normalized gold contained in prediction
    if gold_norm and gold_norm in pred_norm:
        return ScoreResult(True, "contains", gold_norm, pred_norm)

    # 3. Try matching with word→digit direction too
    #    (handles gold="seven", pred contains "7" → both become "seven")
    if gold_word and gold_word in pred_norm:
        return ScoreResult(True, "contains", gold_word, pred_norm)

    return ScoreResult(False, "none", gold_norm, pred_norm)


def is_correct(predicted: str, gold: str) -> bool:
    """Convenience: return True if the predicted answer is correct."""
    return score(predicted, gold).correct
