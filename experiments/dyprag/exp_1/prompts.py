"""
Exp 1 — Oracle Ceiling: Prompt templates for patch generation.

All conditions use the same prompt structure; only the context differs.
The model outputs SEARCH/REPLACE blocks; generate_patches.py converts
them to unified diffs programmatically via difflib.
"""

SYSTEM_PROMPT = """\
You are an expert software engineer. You will be given a bug report describing \
an issue in a Python project. Your task is to generate a minimal fix using \
SEARCH/REPLACE blocks.

Rules:
- The SEARCH section must contain the exact lines from the source file that \
need to be changed (copy them verbatim, including indentation).
- The REPLACE section must contain the replacement lines.
- Make the smallest change necessary to fix the described issue.
- You may use multiple SEARCH/REPLACE blocks if multiple changes are needed.
- Output ONLY the SEARCH/REPLACE blocks. No explanation, no markdown fences.

Format:
<<<< SEARCH
<exact lines from source to find>
====
<replacement lines>
>>>> REPLACE"""


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
        "Output ONLY the blocks using the format shown in the system prompt. "
        "No explanation, no markdown fences."
    )
    return "".join(parts)
