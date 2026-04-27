# marimo-team__marimo easy ledger

Source commit: `979fe6a82ebfbc151de8a59461de5cfafe61db78`

## Overview / Purpose

- [overview_purpose_1] Overview / Purpose (easy) - Explains marimo's purpose as a reactive Python notebook, why it emphasizes reproducibility and app/script reuse, and how its public library supports interactive notebooks. Example questions: What problem is marimo trying to solve compared with traditional notebooks?; How does marimo combine Python notebooks with app deployment?; What kinds of features does the top-level marimo library expose for notebook authors?; Why does marimo emphasize pure Python notebook files?

## Setup and CLI

- [setup_cli_1] Setup and CLI (easy) - Explains how marimo is installed, how the command-line interface introduces editing, running, tutorials, conversion, export, and how the main Click commands route those workflows. Example questions: How does the marimo CLI use tutorials as the first install verification path?; What global options does the root marimo Click group apply before dispatching subcommands?; How does marimo run handle multiple notebook files or directories as an app gallery?; When does marimo export watch mode require an output file?; How are notebook CLI arguments passed through when running or exporting a marimo notebook?

## Public Python Interface

- [public_interface_1] Public Python Interface (easy) - Explains how marimo exposes its top-level Python API, how the API docs group the interface, and when developers should use key public helpers for apps, markdown, state, outputs, and console capture. Example questions: What kinds of features does marimo expose from the top-level mo namespace?; How does the marimo API reference organize public notebook functionality by task?; What is the public markdown workflow for dynamic content and icon rendering in marimo?; How can public marimo helpers make stdout or stderr visible as app cell output?

## Reactive Runtime and Dataflow

- [reactive_runtime_dataflow_1] Reactive Runtime and Dataflow (easy) - Explains how marimo derives a notebook dependency graph from cell definitions and references, how that graph drives reactive execution, stale and disabled behavior, and dataflow visualization. Example questions: How does marimo decide which cells become stale after a cell changes?; What responsibilities does DirectedGraph have in marimo's reactive runtime?; How are disabled cells handled during reactive execution?; What does marimo's dependencies panel reveal about notebook dataflow?

## Apps, Scripts, and WebAssembly

- [apps_scripts_wasm_1] Apps, Scripts, and WebAssembly (easy) - Explains how marimo notebooks are run as web apps, ordinary Python scripts, exported artifacts, and browser-only WebAssembly notebooks. Example questions: How does marimo decide what an app looks like when I use marimo run?; What is different between directly running a marimo notebook as Python and exporting it as a flat script?; Why does a marimo HTML WASM export need to be served over HTTP?; How are package and data-file expectations different for marimo WebAssembly notebooks?; What does the Pyodide bridge let the browser do in an editable WASM notebook?

## UI, Outputs, and Interactivity

- [ui_outputs_interactivity_1] UI, Outputs, and Interactivity (easy) - Explains how marimo connects interactive UI elements, cell reactivity, markdown and rich outputs, and imperative output updates for app-like notebooks. Example questions: How do marimo UI elements send updated values from the browser back to Python?; How does marimo decide what visual output appears for a cell in app view?; How can I embed a marimo slider or other UI element inside markdown?; What is the difference between console output and visual cell output in marimo?; How does marimo turn ordinary Python objects into rich output HTML?

## SQL and Data Integrations

- [sql_data_integrations_1] SQL and Data Integrations (easy) - Explains how marimo SQL cells execute queries, return dataframe-like results, discover database connections and schemas, and connect notebook data workflows to external data sources. Example questions: How does marimo choose the dataframe output type for SQL query results?; What metadata does the Data Sources panel show for detected SQL connections?; How does DuckDB table discovery populate marimo's database and schema models?; How do marimo dataframe tables and charts send selected rows back into Python?; How do the MotherDuck and Google data integration guides fit into marimo's Python-first data workflow?

## AI, MCP, and Agent Pairing

- [ai_mcp_pairing_1] AI, MCP, and Agent Pairing (easy) - Explains how marimo connects its notebook-aware AI assistant, MCP server and client support, external agent pairing, and shared tool registration model. Example questions: How does marimo's AI assistant use live notebook variables as context?; What does marimo expose through its MCP server endpoint?; How do MCP client presets affect the marimo chat panel?; What should a developer do when adding a new marimo AI tool for both backend chat and MCP?; Why does marimo pair recommend agent skills for working with notebooks?
