"""
Exp 1 — Oracle Ceiling: Prompt templates for patch generation.

Condition A uses direct unified-diff output (no source context provided).
Conditions B/C/D use SEARCH/REPLACE blocks over provided source context.
"""

SYSTEM_PROMPT_SEARCH_REPLACE = """\
You are an expert software engineer. You will be given a bug report describing \
an issue in a Python project. Your task is to generate a minimal fix using \
SEARCH/REPLACE blocks.

Rules:
- The SEARCH section must contain the exact lines from the source file that \
need to be changed (copy them verbatim, including indentation).
- The REPLACE section must contain the replacement lines.
- SEARCH and REPLACE must not be identical.
- Use at most 3 SEARCH/REPLACE blocks total.
- Keep each SEARCH section under 40 lines.
- Make the smallest change necessary to fix the described issue.
- Do not rewrite whole files, large import lists, or unrelated functions.
- Output ONLY the SEARCH/REPLACE blocks. No explanation, no markdown fences.

Format:
<<<< SEARCH
<exact lines from source to find>
====
<replacement lines>
>>>> REPLACE"""


SYSTEM_PROMPT_UNIFIED_DIFF = """\
You are an expert software engineer. You will be given a bug report describing \
an issue in a Python project. Your task is to generate a minimal patch.

Rules:
- Output ONLY a valid unified diff patch.
- The patch must begin with lines like: diff --git a/<path> b/<path>
- Include only the files and hunks needed for the fix.
- Minus/plus lines in each hunk must differ (no no-op edits).
- Do not include markdown fences or explanations.
- If you cannot infer a plausible patch from the bug report, output an empty string.
"""


def make_user_prompt(
    problem_statement: str,
    file_content: str | None = None,
    file_path: str | None = None,
) -> str:
    """Build the user message for patch generation.

    Parameters
    ----------
    problem_statement : str
        The SWE-Bench issue / problem statement.
    file_content : str | None
        Source file content (None for condition A).
    file_path : str | None
        Path of the source file within the repo.
    """
    parts = [f"## Bug Report\n\n{problem_statement}"]

    if file_content is not None:
        header = f"## Source File: `{file_path}`" if file_path else "## Source File"
        parts.append(f"\n\n{header}\n\n```python\n{file_content}\n```")

    parts.append(
        "\n\n## Instructions\n\n"
        "Generate SEARCH/REPLACE blocks that fix the bug described above. "
        "Use at most 3 blocks. Keep edits minimal and local. "
        "Output ONLY the blocks using the format shown in the system prompt. "
        "No explanation, no markdown fences."
    )
    return "".join(parts)
