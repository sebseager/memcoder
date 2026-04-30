# Wheels

This directory is reserved for local Python wheel files that are inconvenient to
download or build repeatedly, such as platform-specific ML dependencies.

It is currently empty. Prefer `uv sync` and the indexes configured in
`pyproject.toml` unless a local wheel is explicitly needed for a target machine.
