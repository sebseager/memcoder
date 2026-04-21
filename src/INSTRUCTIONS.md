You are an expert machine learning researcher execting the attached research plan, step by step. Some instructions before we get started:
- Start a `uv` venv environment in this directory (`src`) if it doesn't exist, and use it to manage dependencies and run scripts.
- Complete each stage in its own `stage-{n}` folder inside this directory.
- After completing each stage, write up a `NOTES.md` file in that stage's folder. This is your "lab notebook" for that stage -- add "entries" to it as you go, describing what commands you run, what changes you make, and any observations, key results, and conclusions you draw.
- Prefer small, modular scripts that do one thing well over large, monolithic ones that cover many concerns.
- When producing outputs, output structured data like JSON or CSV, and for things that are interesting to plot, generate the plot in addition to the raw data.
- Re-running scripts should replace data and plots in-place. If you want to keep old versions, timestamp the output directory or filename before re-running.
- If you end up reusing any code from outside the `src` folder, rewrite or copy and modify rather than reference, since code outside `src` is not stable.
