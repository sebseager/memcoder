"""
Exp 1 — Oracle Ceiling: Prompt templates for patch generation.

All conditions use the same prompt structure; only the context differs.
"""

SYSTEM_PROMPT = """\
You are an expert software engineer. You will be given a bug report describing \
an issue in a Python project. Your task is to generate a minimal patch in \
unified diff format that fixes the issue.

Rules:
- Output ONLY the unified diff. No explanation, no markdown fences.
- The diff must start with `--- a/<filepath>` and `+++ b/<filepath>`.
- Use the correct full repository-relative file paths.
- Make the smallest change necessary to fix the described issue.
- Ensure each hunk header (@@ ... @@) has correct line numbers.
- End the patch with a newline character."""


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
        "Generate a unified diff patch that fixes the bug described above. "
        "Output ONLY the patch, starting with `--- a/` and `+++ b/`. "
        "No explanation, no markdown fences. End with a newline."
    )
    return "".join(parts)
