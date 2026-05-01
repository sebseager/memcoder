# Prompt A/B Summary

## Prompts

**baseline** (naive/shine):
> You are a helpful assistant. Answer the question directly and concisely. Output only the final answer.

**detail** (naive/shine):
> You are a helpful assistant. Answer with specific detail — name the identifiers, mechanisms, or steps the question is asking about. If the question asks for multiple items, list each one.

**adapted** (shine only):
> Your weights have been adapted to encode a specific document about a code repository. Draw on that adapted knowledge to answer. You are a helpful assistant. Answer with specific detail — name the identifiers, mechanisms, or steps the question is asking about. If the question asks for multiple items, list each one.

## Cross-repo overview (mean score per cell)

Per-repo means; do **not** average these across repos — different question-style mixes mean cross-repo aggregates can hide flips.

| Repo | Condition | baseline | detail | adapted | Δ detail−baseline | Δ adapted−baseline |
|---|---|---:|---:|---:|---:|---:|
| marimo-team__marimo | naive | 2.25 | 1.79 | — | -0.46 | — |
| marimo-team__marimo | shine | 2.21 | 1.96 | 2.04 | -0.25 | -0.17 |
| antirez__kilo | naive | 2.47 | 2.09 | — | -0.37 | — |
| antirez__kilo | shine | 2.72 | 2.65 | 2.74 | -0.07 | +0.02 |

![cross_repo_deltas](cross_repo_deltas.png)

## marimo-team__marimo

Repo: **marimo-team__marimo**  (n=24 QAs)

- Baseline run: `results/marimo_easy_v0_20260427T1526`
- Detail run:   `results/marimo_easy_v0_detail_20260427T1720`
- Adapted run:  `results/marimo_easy_v0_adapted_20260427T1728`

**Plots:**

![mean_score](marimo-team__marimo/mean_score.png)

![score_distribution](marimo-team__marimo/score_distribution.png)

![per_qa_shine](marimo-team__marimo/per_qa_shine.png)

![mean_score_naive_le2](marimo-team__marimo/mean_score_naive_le2.png)

**Mean judge score (1–5):**

| Condition | baseline | detail | adapted |
|---|---:|---:|---:|
| naive | 2.25 (n=24) | 1.79 (n=24) | — |
| shine | 2.21 (n=24) | 1.96 (n=24) | 2.04 (n=24) |

**Score distributions (% at each score):**

| Cell | 1 | 2 | 3 | 4 | 5 | mean |
|---|---:|---:|---:|---:|---:|---:|
| naive / baseline | 33% | 21% | 38% | 4% | 4% | 2.25 |
| naive / detail | 33% | 54% | 12% | 0% | 0% | 1.79 |
| shine / baseline | 25% | 46% | 12% | 17% | 0% | 2.21 |
| shine / detail | 42% | 29% | 21% | 8% | 0% | 1.96 |
| shine / adapted | 29% | 46% | 17% | 8% | 0% | 2.04 |

**Filtered to questions where baseline-naive ≤ 2** (kept 13/24):

| Cell | mean | n |
|---|---:|---:|
| naive / baseline | 1.38 | 13 |
| naive / detail | 1.54 | 13 |
| shine / baseline | 2.00 | 13 |
| shine / detail | 1.69 | 13 |
| shine / adapted | 1.77 | 13 |

**Per-QA scores:**

| qa_id | b_naive | d_naive | b_shine | d_shine | a_shine |
|---|---:|---:|---:|---:|---:|
| ai_mcp_pairing_ai_mcp_pairing_1_0001 | 1 | 2 | 1 | 2 | 2 |
| ai_mcp_pairing_ai_mcp_pairing_1_0002 | 3 | 2 | 2 | 2 | 2 |
| ai_mcp_pairing_ai_mcp_pairing_1_0003 | 1 | 1 | 3 | 3 | 3 |
| apps_scripts_wasm_apps_scripts_wasm_1_0001 | 1 | 2 | 2 | 2 | 2 |
| apps_scripts_wasm_apps_scripts_wasm_1_0002 | 3 | 2 | 2 | 2 | 2 |
| apps_scripts_wasm_apps_scripts_wasm_1_0003 | 2 | 2 | 1 | 2 | 2 |
| overview_purpose_overview_purpose_1_0001 | 1 | 2 | 1 | 1 | 1 |
| overview_purpose_overview_purpose_1_0002 | 3 | 2 | 2 | 1 | 2 |
| overview_purpose_overview_purpose_1_0003 | 3 | 2 | 2 | 3 | 3 |
| public_interface_public_interface_1_0001 | 3 | 3 | 4 | 3 | 4 |
| public_interface_public_interface_1_0002 | 3 | 1 | 3 | 1 | 2 |
| public_interface_public_interface_1_0003 | 3 | 1 | 1 | 1 | 1 |
| reactive_runtime_dataflow_reactive_runtime_dataflow_1_0001 | 2 | 1 | 2 | 1 | 1 |
| reactive_runtime_dataflow_reactive_runtime_dataflow_1_0002 | 1 | 1 | 4 | 4 | 4 |
| reactive_runtime_dataflow_reactive_runtime_dataflow_1_0003 | 2 | 2 | 2 | 1 | 1 |
| setup_cli_setup_cli_1_0001 | 2 | 2 | 1 | 1 | 1 |
| setup_cli_setup_cli_1_0002 | 2 | 2 | 2 | 1 | 2 |
| setup_cli_setup_cli_1_0003 | 1 | 1 | 2 | 1 | 1 |
| sql_data_integrations_sql_data_integrations_1_0001 | 3 | 3 | 4 | 4 | 3 |
| sql_data_integrations_sql_data_integrations_1_0002 | 3 | 2 | 2 | 3 | 2 |
| sql_data_integrations_sql_data_integrations_1_0003 | 1 | 1 | 4 | 2 | 2 |
| ui_outputs_interactivity_ui_outputs_interactivity_1_0001 | 5 | 3 | 3 | 3 | 3 |
| ui_outputs_interactivity_ui_outputs_interactivity_1_0002 | 1 | 1 | 1 | 1 | 1 |
| ui_outputs_interactivity_ui_outputs_interactivity_1_0003 | 4 | 2 | 2 | 2 | 2 |

<details>
<summary>SHINE answer text across variants</summary>

#### `ai_mcp_pairing_ai_mcp_pairing_1_0001`

**Q:** What safety boundary does marimo's chat panel use when moving from conversation-only help to notebook editing?

**Expected:** The chat panel separates capability by mode: Manual uses only conversation and injected context, Ask adds read-only marimo and MCP client tools for inspection, and Agent adds editing tools that can change or run notebook cells.

- **shine / baseline** (score 1): The safety boundary is the user's notebook variables.
- **shine / detail**   (score 2): Marimo's chat panel uses the notebook scope as a safety boundary when moving from conversation-only help to notebook editing.
- **shine / adapted**  (score 2): Marimo's chat panel uses the user's active notebook context as a safety boundary when moving from conversation-only help to notebook editing.

#### `ai_mcp_pairing_ai_mcp_pairing_1_0002`

**Q:** How does marimo avoid implementing separate tool systems for its chat backend and MCP server?

**Expected:** marimo uses a shared ToolBase architecture and a shared tool registry, so a tool with typed dataclass inputs and outputs can be adapted once for backend chat use and once for MCP registration.

- **shine / baseline** (score 2): Marimo avoids implementing separate tool systems by using a shared notebook state and exposing variables from the running notebook through the chat interface.
- **shine / detail**   (score 2): Marimo avoids implementing separate tool systems for its chat backend and MCP server by using a shared notebook-aware model.
- **shine / adapted**  (score 2): Marimo avoids implementing separate tool systems for its chat backend and MCP server by using a shared notebook state and a shared notebook context. This allows both the chat and MCP server to access and modify the same notebook state, enabling seamless integration and communication between the two components.

#### `ai_mcp_pairing_ai_mcp_pairing_1_0003`

**Q:** What is the specific role of the code-mode MCP server in agent pairing?

**Expected:** The code-mode MCP server exposes session listing and scratchpad execution so an external agent can run temporary code against variables in a live notebook without modifying notebook cells or the dependency graph.

- **shine / baseline** (score 3): The code-mode MCP server is the bridge that connects agent variables to the running notebook, enabling the notebook to access live code state and variables.
- **shine / detail**   (score 3): The code-mode MCP server is the bridge that connects agent variables and user-provided context into a data notebook.
- **shine / adapted**  (score 3): The code-mode MCP server plays a crucial role in agent pairing by providing a bridge between notebook-aware tools and the running notebook. It enables agents to access notebook state and tools inside the notebook's `data mode` context, which is essential for pairing and collaboration.

#### `apps_scripts_wasm_apps_scripts_wasm_1_0001`

**Q:** When marimo runs a folder of notebooks as an app gallery with watch mode, what repository-specific warning does the server implementation surface?

**Expected:** It warns that gallery files are discovered dynamically, so newly created notebooks in that directory may appear in the gallery and execute code when opened; users should rely on trusted directories and authentication controls.

- **shine / baseline** (score 2): The server implementation surfaces a warning that newly added notebook files may appear differently when exported.
- **shine / detail**   (score 2): When marimo runs a folder of notebooks as an app gallery with watch mode, the server implementation surfaces a warning that newly added files may be missing virtual files.
- **shine / adapted**  (score 2): When marimo runs a folder of notebooks as an app gallery with watch mode, the server implementation surfaces a warning that newly added files may be missing virtual files.

#### `apps_scripts_wasm_apps_scripts_wasm_1_0002`

**Q:** What extra work does the HTML WebAssembly CLI export perform around assets and data folders before writing the notebook HTML?

**Expected:** It copies marimo's static assets into the output directory, creates a GitHub Pages compatibility marker, and copies a public folder next to the notebook when one exists.

- **shine / baseline** (score 2): The HTML WebAssembly CLI exports assets and data folders as an application.
- **shine / detail**   (score 2): The HTML WebAssembly CLI exports a default path, merges the notebook and template, applies a marker whose presence is supported by the notebook editor, copies assets and metadata from the notebook and template, strips virtual directories, and optionally strips notebook execution and output when exported differently.
- **shine / adapted**  (score 2): The HTML WebAssembly CLI exports a default path, merges the notebook and session data, applies a marker to the notebook and session data, copies assets and metadata, strips virtual directories, and optionally strips the notebook token.

#### `apps_scripts_wasm_apps_scripts_wasm_1_0003`

**Q:** How does marimo's Pyodide session replace the normal server connection for browser-only notebooks?

**Expected:** It creates browser-callable bridge methods and asynchronous queues for controls, UI/model updates, completions, and input, while streams send kernel and cell notifications back to the frontend.

- **shine / baseline** (score 1): A single notebook becomes an interactive Python program that can be run multiple ways.
- **shine / detail**   (score 2): Marimo's Pyodide session replaces the normal server connection for browser-only notebooks by running Python code in the browser, eliminating the need for a separate server.
- **shine / adapted**  (score 2): Marimo's Pyodide session replaces the normal server connection for browser-only notebooks by running Python code in the browser, eliminating the need for a separate server.

#### `overview_purpose_overview_purpose_1_0001`

**Q:** Which two state-synchronization behaviors does marimo use to prevent notebook state from drifting away from the code?

**Expected:** When a cell runs, marimo reruns dependent cells or marks them stale for expensive notebooks, and when a cell is deleted, marimo removes that cell's variables from program memory.

- **shine / baseline** (score 1): Reactivity and version control
- **shine / detail**   (score 1): Marimo uses the following two state-synchronization behaviors to prevent notebook state from drifting away from the code:
- **shine / adapted**  (score 1): Marimo uses the following two state-synchronization behaviors to prevent notebook state from drifting away from the code:

#### `overview_purpose_overview_purpose_1_0002`

**Q:** How does the overview document connect expensive-notebook behavior with marimo's ordinary-Python reactivity model?

**Expected:** It says expensive-notebook configurations can mark affected cells as stale instead of rerunning them, while reactivity still comes from ordinary Python cells that marimo statically analyzes rather than from special opt-in syntax.

- **shine / baseline** (score 2): The overview document explains that marimo is designed to be reactive, dynamic, and interactive, while also being scriptable, maintainable, and version-controlled.
- **shine / detail**   (score 1): The overview document connects expensive-notebook behavior with marimo's ordinary-Python reactivity model by emphasizing that marimo is designed to be reactive, dynamic, and interactive, while also being scriptable, maintainable, and version-controlled.
- **shine / adapted**  (score 2): The overview document connects expensive-notebook behavior with marimo's ordinary-Python reactivity model by highlighting that marimo is designed to be reactive, dynamic, and interactive, while also being efficient and avoiding the pitfalls of traditional notebooks.

#### `overview_purpose_overview_purpose_1_0003`

**Q:** Why is assigning a marimo UI element to a global variable important?

**Expected:** A UI element assigned to a global variable can synchronize browser interactions back to Python, so cells that depend on that element rerun with its latest value without callback code.

- **shine / baseline** (score 2): Assigning a marimo UI element to a global variable is important because it allows the element to be preserved when the notebook is re-run.
- **shine / detail**   (score 3): Assigning a marimo UI element to a global variable is important because it allows the program to keep track of the marimo UI element and maintain its state.
- **shine / adapted**  (score 3): Assigning a marimo UI element to a global variable is important because it allows the element to be accessed and interacted with from anywhere in the code, making it easier to build interactive and dynamic applications.

#### `public_interface_public_interface_1_0001`

**Q:** How does marimo's root package make a broad implementation feel like a compact public API?

**Expected:** The root package declares the supported exports in marimo/__init__.py and re-exports implementations from internal modules, so users can rely on the top-level mo namespace instead of importing most internals directly.

- **shine / baseline** (score 4): The root package makes a broad implementation feel like a compact public API by exposing the public API of notebook helpers and re-exports implementations from internal modules.
- **shine / detail**   (score 3): The root package makes a broad implementation feel like a compact public API by exposing the public API of marimo as the central namespace.
- **shine / adapted**  (score 4): The root package of marimo makes a broad implementation feel like a compact public API by exposing the public API through `mo` and `marimo`.

#### `public_interface_public_interface_1_0002`

**Q:** According to the public interface document, when should a developer reach for marimo's state API rather than ordinary UI element values?

**Expected:** They should use it only for advanced cases such as preserving history that cannot be computed from a widget's current value, synchronizing multiple UI elements, or introducing cycles across cells. Ordinary interactivity should usually use UI element values and reactive execution.

- **shine / baseline** (score 3): When the UI element is part of a dynamic or complex state flow.
- **shine / detail**   (score 1): The public API is the central declaration of this surface: it names the supported inputs and outputs, defines the interface, and shows the output helpers.
- **shine / adapted**  (score 2): The public interface document suggests that developers should reach for marimo's state API rather than ordinary UI element values when they need to work with the dynamic and reactive nature of the framework, such as managing complex state interactions, handling user input, or accessing internal framework state.

#### `public_interface_public_interface_1_0003`

**Q:** What happens if a marimo cell both writes output programmatically and ends with a non-empty expression, and what should the developer do to add output instead?

**Expected:** The non-empty last expression replaces earlier output. To add to existing output instead of replacing it, the developer should use the public append output helper.

- **shine / baseline** (score 1): The developer should replace the existing code with `print` and `marimo.display` to add output instead.
- **shine / detail**   (score 1): If a marimo cell both writes output programmatically and ends with a non-empty expression, the developer should replace the existing expression with `print` instead of `marimo` so that the output is displayed in the notebook rather than in a separate cell.
- **shine / adapted**  (score 1): If a marimo cell both writes output programmatically and ends with a non-empty expression, the developer should use `print` for output instead.

#### `reactive_runtime_dataflow_reactive_runtime_dataflow_1_0001`

**Q:** What sequence does the document describe when `DirectedGraph` registers a new cell?

**Expected:** Registration records the cell's definitions, computes parent and child edges from definitions and references, adds those edges to the topology, and asks the cycle tracker whether any new edge closes a path back to its source.

- **shine / baseline** (score 2): The sequence is: `DirectedGraph` reads the cell's program, builds a graph, computes its top order, and computes the dependencies of cells that reference it.
- **shine / detail**   (score 1): The document describes the following sequence when `DirectedGraph` registers a new cell:
- **shine / adapted**  (score 1): The document describes the following sequence when `DirectedGraph` registers a new cell:

#### `reactive_runtime_dataflow_reactive_runtime_dataflow_1_0002`

**Q:** When the notebook runner is asked to run root cells, what extra cells can it add to the execution queue?

**Expected:** It always adds stale ancestors of the requested roots so upstream values are current, and in autorun mode it also adds descendants that are affected by those roots before topologically sorting the selected cells.

- **shine / baseline** (score 4): The notebook runner can add parent and child cells to the execution queue.
- **shine / detail**   (score 4): The notebook runner can add parent and child cells to the execution queue.
- **shine / adapted**  (score 4): The notebook runner can add parent and child cells to the execution queue.

#### `reactive_runtime_dataflow_reactive_runtime_dataflow_1_0003`

**Q:** Why is downstream tracing in marimo's minimap broader than upstream tracing?

**Expected:** A marimo cell can define multiple variables, so upstream tracing can identify exactly which variables a selected cell reads, but downstream tracing treats a dependent cell as affected if it references any definition from the selected cell.

- **shine / baseline** (score 2): Downstream tracing is broader than upstream tracing because marimo's reactive runtime is a directed graph, not a top-down program.
- **shine / detail**   (score 1): Marimo's reactive runtime is built around the idea that a notebook is a directed graph of cells, not a top-down program.
- **shine / adapted**  (score 1): Marimo's reactive runtime is built around the idea that a notebook is a directed graph of cells, not a top-down program.

#### `setup_cli_setup_cli_1_0001`

**Q:** What first-time-user actions does marimo's root CLI help emphasize instead of only listing commands mechanically?

**Expected:** It highlights launching the intro tutorial, editing a new notebook, running a notebook as an app, and asking the tutorial command for its own help.

- **shine / baseline** (score 1): The root CLI emphasizes first-time-user actions such as setting up the environment, installing dependencies, and configuring the workspace.
- **shine / detail**   (score 1): Marimo's root CLI helps emphasize first-time-user actions such as setting up the working environment, installing dependencies, and configuring the editor.
- **shine / adapted**  (score 1): Marimo's root CLI emphasizes the first-time-user experience by highlighting the following actions:

#### `setup_cli_setup_cli_1_0002`

**Q:** How does the edit command decide what notebook or directory to open when started from the CLI?

**Expected:** It can use a supplied filename, directory, validated URL, or piped notebook contents on Unix-like systems; if no name is supplied, it opens the current working directory, and missing notebook files are created as empty files.

- **shine / baseline** (score 2): The edit command decides what notebook or directory to open based on the current working directory.
- **shine / detail**   (score 1): The edit command decides what notebook or directory to open when started from the CLI by launching the intro session.
- **shine / adapted**  (score 2): The edit command decides what notebook or directory to open when started from the CLI by launching the current working directory.

#### `setup_cli_setup_cli_1_0003`

**Q:** Which export paths run a marimo app before writing output, and what restriction does watch mode have?

**Expected:** HTML exports and output-including Jupyter exports run the app first so notebook arguments can affect results. Watch mode is supported only when an output file is provided.

- **shine / baseline** (score 2): The export paths run a marimo app before writing output, and watch mode is restricted to read-only mode.
- **shine / detail**   (score 1): The export paths that run a marimo app before writing output are `marimo/_cli/cli.py` and `marimo/_cli/cli_jupyter.py`.
- **shine / adapted**  (score 1): The export paths that run a marimo app before writing output are `marimo/_cli/cli.py` and `marimo/_cli/cli_jupyter.py`.

#### `sql_data_integrations_sql_data_integrations_1_0001`

**Q:** Why are marimo SQL cells described as part of the Python notebook model instead of a separate SQL-only layer?

**Expected:** A SQL cell is syntactic sugar for a Python cell that calls `mo.sql`, so the notebook stays a pure Python script. The SQL is represented as a formatted Python string, letting queries depend on UI controls and other Python values in marimo's reactive graph.

- **shine / baseline** (score 4): Marimo SQL cells are described as part of the Python notebook model because they are syntactically and semantically equivalent to Python cells with `mo.sql`, allowing users to write SQL directly in the same environment as Python code.
- **shine / detail**   (score 4): Marimo SQL cells are described as part of the Python notebook model because they are syntactically and semantically equivalent to `mo.sql`, which is a Python cell that wraps `duckdb`. This means that marimo SQL cells are treated as Python code within the notebook, allowing for seamless integration with the rest of the notebook's features and execution flow.
- **shine / adapted**  (score 3): The marimo SQL cells are described as part of the Python notebook model rather than a separate SQL-only layer because they are integrated within the same environment and share the same UI, metadata, and execution context as Python cells. This allows users to seamlessly switch between Python and SQL cells, maintain a consistent workflow, and leverage the same notebook interface for both types of cells.

#### `sql_data_integrations_sql_data_integrations_1_0002`

**Q:** How does marimo make Python-defined database connections available to SQL cells?

**Expected:** It scans notebook variables for values compatible with its supported engine classes, wraps matching values as query engines or catalog-aware connections, and then exposes those compatible Python connection variables as selectable SQL connections in the notebook.

- **shine / baseline** (score 2): Marimo makes Python-defined database connections available to SQL cells as a lazy Python object.
- **shine / detail**   (score 3): Marimo makes Python-defined database connections available to SQL cells by treating them as part of the same reactive notebook model.
- **shine / adapted**  (score 2): Marimo makes Python-defined database connections available to SQL cells by treating SQL as part of the same reactive Python model.

#### `sql_data_integrations_sql_data_integrations_1_0003`

**Q:** What does marimo do when datasource introspection fails for a detected connection?

**Expected:** It logs a warning and still returns a connection entry for the Data Sources panel instead of crashing the notebook, leaving any unavailable database, schema, table, or column details empty.

- **shine / baseline** (score 4): Marimo returns a warning compatible with the UI when datasource introspection fails for a detected connection.
- **shine / detail**   (score 2): When datasource introspection fails for a detected connection, marimo defaults to a Python helper object in the editor.
- **shine / adapted**  (score 2): When datasource introspection fails for a detected connection, marimo falls back to a default database connection in the editor.

#### `ui_outputs_interactivity_ui_outputs_interactivity_1_0001`

**Q:** Why does marimo require an interacted UI element to be bound to a global variable for notebook reactivity to occur?

**Expected:** marimo's interaction rule reruns cells that reference the global variable holding the UI element without defining it. Without that global binding, the browser value can update the element object, but marimo has no variable dependency to use for rerunning other cells.

- **shine / baseline** (score 3): The interaction output is bound to a global variable for notebook reactivity to occur.
- **shine / detail**   (score 3): Marimo makes the interaction output the last expression in the cell, so a UI element must be bound to a global variable for notebook reactivity to occur.
- **shine / adapted**  (score 3): Marimo requires an interacted UI element to be bound to a global variable for notebook reactivity to occur because the UI element must be accessible to the global variable that is used as the notebook's interface. This allows the notebook to react to changes in the UI element's state.

#### `ui_outputs_interactivity_ui_outputs_interactivity_1_0002`

**Q:** What two identity mechanisms does a marimo UI element use, and what is each for?

**Expected:** A stable object id synchronizes browser instances of the element with the Python kernel. A separate random token changes whenever the creating cell reruns so the frontend resets to the element's new initial value.

- **shine / baseline** (score 1): A marimo UI element uses a visual element and a layout object. The visual element is the UI object shown above the cell, and the layout object is the UI object returned under `marimo.ui` and is used as the element's interface.
- **shine / detail**   (score 1): A marimo UI element uses `marimo.ui.output` for the visual output and `marimo.ui.fragment` for the layout object.
- **shine / adapted**  (score 1): A marimo UI element uses `mo.md` and `mo.ui` as its two identity mechanisms. `mo.md` is the markdown representation of the element, while `mo.ui` is the UI element that appears alongside it.

#### `ui_outputs_interactivity_ui_outputs_interactivity_1_0003`

**Q:** What happens if a cell both imperatively appends output and then ends with a non-None last expression?

**Expected:** The non-None last expression replaces the existing visual output, including anything previously written imperatively, unless the author wraps that final value in an append operation.

- **shine / baseline** (score 2): The last expression is not included in the output.
- **shine / detail**   (score 2): When a cell both imperatively appends output and then ends with a non-None last expression, the last expression is not included in the output.
- **shine / adapted**  (score 2): When a cell both imperatively appends output and ends with a non-None last expression, the last expression is not included in the output.

</details>

## antirez__kilo

Repo: **antirez__kilo**  (n=43 QAs)

- Baseline run: `results/kilo_easy_v0_20260427T0138`
- Detail run:   `results/kilo_easy_v0_detail_20260427T1731`
- Adapted run:  `results/kilo_easy_v0_adapted_20260427T1747`

**Plots:**

![mean_score](antirez__kilo/mean_score.png)

![score_distribution](antirez__kilo/score_distribution.png)

![per_qa_shine](antirez__kilo/per_qa_shine.png)

![mean_score_naive_le2](antirez__kilo/mean_score_naive_le2.png)

**Mean judge score (1–5):**

| Condition | baseline | detail | adapted |
|---|---:|---:|---:|
| naive | 2.47 (n=43) | 2.09 (n=43) | — |
| shine | 2.72 (n=43) | 2.65 (n=43) | 2.74 (n=43) |

**Score distributions (% at each score):**

| Cell | 1 | 2 | 3 | 4 | 5 | mean |
|---|---:|---:|---:|---:|---:|---:|
| naive / baseline | 21% | 42% | 19% | 7% | 12% | 2.47 |
| naive / detail | 44% | 16% | 30% | 5% | 5% | 2.09 |
| shine / baseline | 16% | 30% | 26% | 21% | 7% | 2.72 |
| shine / detail | 30% | 19% | 21% | 16% | 14% | 2.65 |
| shine / adapted | 28% | 26% | 7% | 23% | 16% | 2.74 |

**Filtered to questions where baseline-naive ≤ 2** (kept 27/43):

| Cell | mean | n |
|---|---:|---:|
| naive / baseline | 1.67 | 27 |
| naive / detail | 1.70 | 27 |
| shine / baseline | 2.59 | 27 |
| shine / detail | 2.37 | 27 |
| shine / adapted | 2.44 | 27 |

**Per-QA scores:**

| qa_id | b_naive | d_naive | b_shine | d_shine | a_shine |
|---|---:|---:|---:|---:|---:|
| overview_purpose_overview_purpose_1_0001 | 1 | 1 | 5 | 5 | 5 |
| overview_purpose_overview_purpose_1_0002 | 1 | 2 | 2 | 1 | 1 |
| overview_purpose_overview_purpose_1_0003 | 3 | 4 | 5 | 5 | 5 |
| overview_purpose_overview_purpose_1_0004 | 4 | 1 | 4 | 3 | 2 |
| overview_purpose_overview_purpose_1_0005 | 2 | 2 | 4 | 1 | 1 |
| overview_purpose_overview_purpose_1_0006 | 1 | 1 | 2 | 2 | 1 |
| overview_purpose_overview_purpose_1_0007 | 2 | 2 | 4 | 1 | 4 |
| overview_purpose_overview_purpose_1_0008 | 3 | 1 | 3 | 3 | 4 |
| overview_purpose_overview_purpose_1_0009 | 3 | 3 | 3 | 1 | 1 |
| overview_purpose_overview_purpose_1_0010 | 1 | 2 | 3 | 3 | 3 |
| overview_purpose_overview_purpose_1_0011 | 2 | 1 | 2 | 4 | 4 |
| overview_purpose_overview_purpose_1_0012 | 2 | 1 | 1 | 2 | 2 |
| overview_purpose_overview_purpose_1_0013 | 2 | 3 | 3 | 2 | 2 |
| overview_purpose_overview_purpose_1_0014 | 3 | 3 | 3 | 4 | 4 |
| overview_purpose_overview_purpose_1_0015 | 2 | 1 | 2 | 4 | 4 |
| overview_purpose_overview_purpose_1_0016 | 2 | 1 | 3 | 3 | 2 |
| raw_terminal_input_1_0001 | 5 | 3 | 4 | 5 | 5 |
| raw_terminal_input_1_0002 | 3 | 3 | 1 | 4 | 5 |
| raw_terminal_input_1_0003 | 1 | 1 | 5 | 5 | 5 |
| raw_terminal_input_1_0004 | 5 | 5 | 2 | 1 | 1 |
| raw_terminal_input_1_0005 | 2 | 3 | 3 | 1 | 1 |
| raw_terminal_input_1_0006 | 4 | 3 | 4 | 5 | 5 |
| raw_terminal_input_1_0007 | 2 | 4 | 4 | 3 | 4 |
| raw_terminal_input_1_0008 | 2 | 1 | 2 | 2 | 2 |
| raw_terminal_input_1_0009 | 2 | 1 | 2 | 2 | 1 |
| raw_terminal_input_1_0010 | 3 | 3 | 3 | 3 | 3 |
| raw_terminal_input_1_0011 | 3 | 2 | 1 | 1 | 1 |
| raw_terminal_input_1_0012 | 2 | 1 | 2 | 1 | 1 |
| raw_terminal_input_1_0013 | 5 | 3 | 4 | 5 | 4 |
| raw_terminal_input_1_0014 | 1 | 3 | 3 | 1 | 2 |
| rows_editing_persistence_1_0001 | 2 | 1 | 3 | 3 | 3 |
| rows_editing_persistence_1_0002 | 4 | 3 | 3 | 3 | 4 |
| rows_editing_persistence_1_0003 | 1 | 3 | 4 | 4 | 4 |
| rows_editing_persistence_1_0004 | 1 | 1 | 2 | 4 | 4 |
| rows_editing_persistence_1_0005 | 2 | 2 | 1 | 3 | 2 |
| rows_editing_persistence_1_0006 | 3 | 1 | 1 | 1 | 2 |
| rows_editing_persistence_1_0007 | 2 | 2 | 1 | 2 | 1 |
| rows_editing_persistence_1_0008 | 2 | 3 | 2 | 1 | 2 |
| rows_editing_persistence_1_0009 | 5 | 5 | 2 | 4 | 5 |
| rows_editing_persistence_1_0010 | 2 | 1 | 2 | 2 | 2 |
| rows_editing_persistence_1_0011 | 2 | 1 | 2 | 1 | 2 |
| rows_editing_persistence_1_0012 | 5 | 1 | 4 | 2 | 1 |
| rows_editing_persistence_1_0013 | 1 | 1 | 1 | 1 | 1 |

<details>
<summary>SHINE answer text across variants</summary>

#### `overview_purpose_overview_purpose_1_0001`

**Q:** What is kilo and what is its intended purpose?

**Expected:** Kilo is a very small terminal text editor in under 1K lines of C, intended as a simple, hackable starting point for building more advanced editors or terminal interfaces.

- **shine / baseline** (score 5): Kilo is a small, single-file software editor meant to provide basic editing in less than one thousand lines of C while remaining easy to read and hack.
- **shine / detail**   (score 5): Kilo is a very small single-file software editor meant to demonstrate the essential pieces of a terminal editor in less than one thousand lines of C while remaining easy to read and hack.
- **shine / adapted**  (score 5): Kilo is a very small single-file software editor, also known as antirez. Its intended purpose is to demonstrate the essential pieces of a terminal editor in less than one thousand lines of C while remaining easy to read and hack.

#### `overview_purpose_overview_purpose_1_0002`

**Q:** Which three keyboard shortcuts are emphasized in kilo's overview?

**Expected:** Ctrl-S saves, Ctrl-Q quits, and Ctrl-F searches.

- **shine / baseline** (score 2): Ctrl-S
- **shine / detail**   (score 1): Kilo is a very small single-file editor, also known as antiloop. It is intended to demonstrate the essential pieces of a terminal editor: cursor position, scroll offsets, an editable in-memory row model, file persistence, status, refresh, and selected text.
- **shine / adapted**  (score 1): Kilo's overview emphasizes the essential single-keypress C, Q, and ESC.

#### `overview_purpose_overview_purpose_1_0003`

**Q:** How is kilo invoked from the command line?

**Expected:** It is started as `kilo <filename>`.

- **shine / baseline** (score 5): Kilo is invoked as `kilo <filename>`.
- **shine / detail**   (score 5): Kilo is invoked as `kilo <filename>`.
- **shine / adapted**  (score 5): Kilo is invoked as `kilo <filename>`.

#### `overview_purpose_overview_purpose_1_0004`

**Q:** Does kilo use curses or another terminal UI library?

**Expected:** No. It is self-contained and talks directly to the terminal with fairly standard VT100-style escape sequences.

- **shine / baseline** (score 4): No, kilo does not use curses or another terminal UI library.
- **shine / detail**   (score 3): Kilo is a single-file, text-mode editor written in less than one thousand lines of C, and relies on `termios` and `linenoise` libraries, rather than curses.
- **shine / adapted**  (score 2): Kilo is a single-file, text-mode editor written in less than one thousand lines of C, meant to demonstrate the essential pieces of a terminal editor: raw user input, scrolling with cursor offsets, an editable in-memory row model, file persistence, status UI, search, and syntax highlighting.

#### `overview_purpose_overview_purpose_1_0005`

**Q:** What does kilo do at startup before entering its main loop?

**Expected:** It initializes global editor state, selects syntax highlighting based on the filename, opens the requested file, enables raw terminal mode, and sets a help message.

- **shine / baseline** (score 4): Kilo initializes the global editor state, reads the target file, builds an in-memory representation, and sets a temporary terminal state.
- **shine / detail**   (score 1): Kilo is a very small single-file editor written in less than one thousand lines of C, meant to demonstrate the essential pieces of a terminal editor: raw user input, scrolling with cursor offsets, an editable in-memory row model, file persistence, status UI, search, and syntax highlighting.
- **shine / adapted**  (score 1): Kilo does not perform any actions at startup.

#### `overview_purpose_overview_purpose_1_0006`

**Q:** What is the core loop of kilo's architecture?

**Expected:** It loops forever refreshing the screen and processing one keypress at a time.

- **shine / baseline** (score 2): The core loop of kilo's architecture is a single `main` function.
- **shine / detail**   (score 2): Kilo's core loop is a single `main` function with a few `struct` and `linenoise` C model81 line editor used to read the target text, a renderer, and syntax-highlighting library selected in the Makefile.
- **shine / adapted**  (score 1): Kilo's core architecture is built in a few hours from ideas in the author's `kilo.c` and `linenoise` projects, and its main value is educational and hackable rather than search-match.

#### `overview_purpose_overview_purpose_1_0007`

**Q:** Which global structure holds most of kilo's editor state?

**Expected:** Most editor state lives in the global `struct editorConfig E`.

- **shine / baseline** (score 4): The global `struct editorConfig` holds most of kilo's editor state.
- **shine / detail**   (score 1): Kilo is a very small single-file software editor, authoring less than one thousand lines of C while remaining easy to read and hack. Its purpose is to demonstrate the essential pieces of a terminal editor: raw user input, scrolling with one keypress, an editable in-memory row model, file persistence, status UI, search, and syntax highlighting.
- **shine / adapted**  (score 4): Kilo's editor state is stored in a single `struct editorConfig`

#### `overview_purpose_overview_purpose_1_0008`

**Q:** What kinds of information are stored in `struct editorConfig E`?

**Expected:** It stores cursor position, scroll offsets, terminal dimensions, file rows, dirty state, filename, status message, and selected syntax.

- **shine / baseline** (score 3): The `struct editorConfig E` stores information such as cursor position, line length, scrolling state, file type, status, search text, and selected flags.
- **shine / detail**   (score 3): `struct editorConfig E` stores the global editor state, including cursor position, scroll offsets, terminal dimensions, file size, search text, and selected flags.
- **shine / adapted**  (score 4): `struct editorConfig E` stores the global editor state, including cursor position, line and row lengths, window dimensions, scroll offsets, filename, status message, selected file, search text, and search state.

#### `overview_purpose_overview_purpose_1_0009`

**Q:** What does an `erow` represent in kilo?

**Expected:** An `erow` represents one file line and stores the original characters, a rendered version used for display, and syntax-highlight metadata.

- **shine / baseline** (score 3): An `erow` represents one line in the text being edited.
- **shine / detail**   (score 1): Kilo is a very small single-file editor written in less than one thousand lines of C, meant to demonstrate the essential pieces of a terminal editor: raw user input, scrolling with cursor offsets, an editable in-memory row model, file persistence, status UI, search, and syntax highlighting.
- **shine / adapted**  (score 1): An `erow` represents a single character in kilo.

#### `overview_purpose_overview_purpose_1_0010`

**Q:** What are kilo's core user-facing features?

**Expected:** Its core user-facing features are basic file editing, saving, quitting, searching, scrolling, and C/C++ syntax highlighting.

- **shine / baseline** (score 3): Kilo's core user-facing features are basic editor, cursor, and syntax highlighting.
- **shine / detail**   (score 3): Kilo's core user-facing features are basic editor, scroll, and highlight.
- **shine / adapted**  (score 3): Kilo's core user-facing features are basic editor, scroll, and highlight.

#### `overview_purpose_overview_purpose_1_0011`

**Q:** How does kilo load a target file into memory?

**Expected:** `editorOpen` reads the target file line by line with `getline`, strips trailing newline or carriage-return characters, and inserts each line into the in-memory row array.

- **shine / baseline** (score 2): Kilo loads a target file into memory by reading the file with `fopen` and `fgets`.
- **shine / detail**   (score 4): Kilo loads a target file into memory by reading the target file line by line with `getline()`.
- **shine / adapted**  (score 4): Kilo loads a target file into memory by reading the target file line by line with `getline()`.

#### `overview_purpose_overview_purpose_1_0012`

**Q:** What happens if the requested file does not exist?

**Expected:** Opening a nonexistent file is treated as a new empty file rather than an error.

- **shine / baseline** (score 1): The program terminates.
- **shine / detail**   (score 2): If the requested file does not exist, Kilo prints an error message.
- **shine / adapted**  (score 2): If the requested file does not exist, Kilo prints an error message.

#### `overview_purpose_overview_purpose_1_0013`

**Q:** How does kilo save the current buffer?

**Expected:** `editorSave` converts the row array into one buffer, opens or creates the target file, truncates it to the new length, writes the buffer, clears the dirty flag on success, and reports status.

- **shine / baseline** (score 3): Kilo saves the current buffer by writing it to the file with the same name.
- **shine / detail**   (score 2): Kilo saves the current buffer by writing it to a file with the filename `kilo.c`.
- **shine / adapted**  (score 2): Kilo saves the current buffer by writing it to a file with the filename and metadata.

#### `overview_purpose_overview_purpose_1_0014`

**Q:** Why does kilo build a temporary append buffer while rendering?

**Expected:** It builds a temporary append buffer of terminal escape sequences and visible text so it can write the screen to standard output in one call and reduce flicker.

- **shine / baseline** (score 3): To reduce flickering.
- **shine / detail**   (score 4): Kilo builds a temporary append buffer while rendering to reduce flickering and improve performance.
- **shine / adapted**  (score 4): Kilo builds a temporary append buffer while rendering to reduce flickering and improve performance.

#### `overview_purpose_overview_purpose_1_0015`

**Q:** What does interactive search do as the user types a query?

**Expected:** It updates the status line, refreshes the screen, searches rendered rows, scrolls to the current match, and temporarily marks matched text with `HL_MATCH`.

- **shine / baseline** (score 2): As the user types a query, interactive search builds a temporary editor state, selects a cursor position, and highlights the current buffer line with an editable in-line UI.
- **shine / detail**   (score 4): As the user types a query, interactive search builds a temporary editor state, scrolls the screen with cursor offsets, searches through the current buffer, and highlights each match.
- **shine / adapted**  (score 4): As the user types a query, interactive search builds a temporary editor state, scrolls the screen with cursor offsets, searches through the current buffer, and highlights each match.

#### `overview_purpose_overview_purpose_1_0016`

**Q:** What happens when the user tries to quit with unsaved changes?

**Expected:** Kilo requires repeated Ctrl-Q presses before quitting when the file has unsaved changes.

- **shine / baseline** (score 3): The user is prompted to save the current buffer.
- **shine / detail**   (score 3): If the user tries to quit with unsaved changes, the program will display a warning message.
- **shine / adapted**  (score 2): When the user tries to quit with unsaved changes, the editor warns the user and then shows a help message.

#### `raw_terminal_input_1_0001`

**Q:** Why does kilo handle terminal input itself instead of using curses?

**Expected:** Kilo handles terminal input itself rather than relying on curses or another UI library, so it stays self-contained.

- **shine / baseline** (score 4): Kilo handles terminal input itself rather than relying on curses because it requires raw input and sets up the terminal in a state that allows for precise control over input handling.
- **shine / detail**   (score 5): Kilo handles terminal input itself rather than relying on curses because it needs to precisely control the terminal and manage raw input directly.
- **shine / adapted**  (score 5): Kilo handles terminal input itself rather than relying on curses because it is designed to be a minimalistic and customizable text-based editor that does not require the additional complexity and dependencies of a full-featured library like curses.

#### `raw_terminal_input_1_0002`

**Q:** What does `enableRawMode` do at startup?

**Expected:** It captures the current terminal settings, registers a cleanup hook so they can be restored on exit, and applies a modified attribute set tuned for byte-by-byte interactive editing.

- **shine / baseline** (score 1): EnableRawMode
- **shine / detail**   (score 4): At startup, `enableRawMode` captures the current settings, sets raw mode, and applies a cleanup hook so the cursor can be restored if the process terminates.
- **shine / adapted**  (score 5): At startup, `enableRawMode` captures the current settings, registers a cleanup hook, and applies a modified attribute set tuned for byte-by-byte interactive editing.

#### `raw_terminal_input_1_0003`

**Q:** Where are the original terminal settings kept so they can be restored later?

**Expected:** They are kept in a file-scope variable so the cleanup hook has them when the process ends.

- **shine / baseline** (score 5): The original terminal settings are kept in a file-scope variable so they can be restored later.
- **shine / detail**   (score 5): The original terminal settings are kept in a file-scope variable so they can be restored later.
- **shine / adapted**  (score 5): The original settings are stored in a file-scope variable so they can be restored if the process is killed or if an error occurs.

#### `raw_terminal_input_1_0004`

**Q:** What does `enableRawMode` do if raw mode is already active?

**Expected:** It returns successfully without touching the terminal again.

- **shine / baseline** (score 2): If raw mode is already active, `enableRawMode` returns `true`.
- **shine / detail**   (score 1): If raw mode is already active, `enableRawMode` sets `errno` to `EAGAIN` and returns `-1`.
- **shine / adapted**  (score 1): If raw mode is already active, `enableRawMode` returns `fail`.

#### `raw_terminal_input_1_0005`

**Q:** When does `enableRawMode` take its fatal path, and what does it do?

**Expected:** When standard input is not a TTY, or when the system calls for reading or writing the terminal attributes fail. It sets `errno` and returns `-1`.

- **shine / baseline** (score 3): `enableRawMode` takes its fatal path if raw input is not a TTY, and it sets `errno` to `ENOTTY` and returns `-1`.
- **shine / detail**   (score 1): `enableRawMode` takes its fatal path if `getCursorPosition` returns an error.
- **shine / adapted**  (score 1): `enableRawMode` takes its fatal path if `getSettings` has already been called.

#### `raw_terminal_input_1_0006`

**Q:** What kinds of line-based comforts does raw mode disable so each keypress arrives as a raw byte?

**Expected:** Echo, canonical line buffering, signal generation from keys like Ctrl-C, and extended input processing are all disabled.

- **shine / baseline** (score 4): Echo, canonical line editing, and signal handling
- **shine / detail**   (score 5): Raw mode disables line-based comforts such as echo, canonical line buffering, signal handling, and extended input processing.
- **shine / adapted**  (score 5): Raw mode disables line-based comforts such as echo, canonical line buffering, signal handling, and extended input processing.

#### `raw_terminal_input_1_0007`

**Q:** Why is output post-processing turned off in raw mode?

**Expected:** So kilo can emit precise control sequences without the terminal modifying them.

- **shine / baseline** (score 4): Output post-processing is turned off in raw mode to enable precise control over terminal input and output.
- **shine / detail**   (score 3): In raw mode, output post-processing is turned off to enable precise control over terminal input and output.
- **shine / adapted**  (score 4): In raw mode, output post-processing is turned off to allow for precise control over the terminal's behavior and to enable features like raw input and output, which are essential for certain applications such as interactive programs or low-level terminal manipulation.

#### `raw_terminal_input_1_0008`

**Q:** How is read latency tuned for an interactive editor?

**Expected:** The terminal is configured so that a read can return after roughly a tenth of a second when no byte is available, rather than blocking.

- **shine / baseline** (score 2): Read latency is tuned for an interactive editor by setting `enableRawInputMode` to `true`.
- **shine / detail**   (score 2): Read latency is tuned for an interactive editor by enabling `enableRawInput` and setting `editorReadKey`, which captures the current settings, stores them in a file-scope variable, and applies them on exit.
- **shine / adapted**  (score 2): Read latency is tuned for an interactive editor by enabling `enableRawInput` and setting `editorReadKey`.

#### `raw_terminal_input_1_0009`

**Q:** What does `editorReadKey` do?

**Expected:** It keeps polling for an initial byte and exits the editor if the underlying read reports an error.

- **shine / baseline** (score 2): `editorReadKey` captures the current input and sets up a cleanup hook so the editor can emit precise line-by-line terminal events.
- **shine / detail**   (score 2): `editorReadKey` captures the current state of the terminal.
- **shine / adapted**  (score 1): `editorReadKey` captures the current settings, stores them in a file-scope variable, applies a cleanup hook so they can be restored if the process is killed, and returns a modified input byte array tuned for byte-by-byte interactive editing.

#### `raw_terminal_input_1_0010`

**Q:** How does kilo handle terminal escape sequences for special keys?

**Expected:** It decodes the trailing bytes and returns an internal soft key code rather than the raw escape characters.

- **shine / baseline** (score 3): Kilo captures terminal escape sequences, stores them in a state byte, and applies a modified read-byte routine.
- **shine / detail**   (score 3): Kilo handles terminal escape sequences for special keys by enabling raw input mode, which captures the escape sequences so they can be processed by the editor.
- **shine / adapted**  (score 3): Kilo handles terminal escape sequences for special keys by capturing the escape sequence, storing it in a state variable, and applying a modified read-byte trick.

#### `raw_terminal_input_1_0011`

**Q:** Which keys does kilo recognize through escape sequences?

**Expected:** Bare escapes, the four arrow keys, page up and page down, home and end, and the delete key.

- **shine / baseline** (score 1): Kilo recognizes the following keys through escape sequences:
- **shine / detail**   (score 1): Kilo recognizes the following keys through escape sequences:
- **shine / adapted**  (score 1): Kilo recognizes terminal input through `raw mode`.

#### `raw_terminal_input_1_0012`

**Q:** What does kilo do when the standard ioctl-based window-size query does not yield a column count?

**Expected:** It falls back to a cursor-query trick: it parks the cursor in the bottom-right corner, asks the terminal where the cursor ended up via `getCursorPosition`, and parses the row and column out of the reply.

- **shine / baseline** (score 2): Kilo falls back to a raw read from the terminal.
- **shine / detail**   (score 1): Kilo escapes the shell so they can emit precise control sequences.
- **shine / adapted**  (score 1): Kilo falls back on a read-scan trick: it stores the underlying bytes in a file-scope variable, then applies a modified read and write message.

#### `raw_terminal_input_1_0013`

**Q:** Why does kilo reserve two rows at the bottom of the terminal?

**Expected:** For its status line and message area.

- **shine / baseline** (score 4): Kilo reserves two rows at the bottom of the terminal for status and error messages.
- **shine / detail**   (score 5): Kilo reserves two rows at the bottom of the terminal for status and error messages.
- **shine / adapted**  (score 4): Kilo reserves two rows at the bottom of the terminal for status and cursor line information.

#### `raw_terminal_input_1_0014`

**Q:** How does kilo react to terminal resize events?

**Expected:** A `SIGWINCH` handler updates the stored dimensions, clamps the cursor if it now sits outside the screen, and refreshes the display so the user sees the new layout immediately.

- **shine / baseline** (score 3): Kilo handles terminal resize events by updating the terminal state and reflowing the display.
- **shine / detail**   (score 1): Kilo handles terminal resize events by capturing the current settings, storing them in a cleanup hook, and then applying the stored settings after the resize.
- **shine / adapted**  (score 2): Kilo handles terminal resize events by capturing the current settings, storing them in a cleanup hook, and then applying the stored settings after the resize.

#### `rows_editing_persistence_1_0001`

**Q:** How does kilo represent the editable file in memory?

**Expected:** As an in-memory array of `erow` structures, each carrying the original characters of the line, a parallel rendered version, syntax-highlight metadata, and a small flag tracking whether the previous line ended inside a multiline comment.

- **shine / baseline** (score 3): Kilo represents the editable file as an in-memory array of `erow` structures.
- **shine / detail**   (score 3): Kilo represents the editable file in memory as an in-memory copy of the file, a parallel rendered version used for display, syntax highlighting metadata, and a small flag tracking whether the previous buffer ended with a newline.
- **shine / adapted**  (score 3): Kilo represents the editable file in memory as an in-memory copy of the file, a parallel rendered version used for display, a small block of metadata, and a buffer tracking whether the previous buffer ended inside a paragraph.

#### `rows_editing_persistence_1_0002`

**Q:** Why does each row store both an original character buffer and a rendered version?

**Expected:** Because what shows on screen is not always the same as what is on disk; the rendered view exists for display.

- **shine / baseline** (score 3): To display the visible characters and highlight the current cursor position.
- **shine / detail**   (score 3): Each row stores both an original character buffer and a rendered version to support features like syntax highlighting, highlighting the current block under the cursor, and line numbers.
- **shine / adapted**  (score 4): Each row stores both an original character buffer and a rendered version to support editing and display functionalities.

#### `rows_editing_persistence_1_0003`

**Q:** How are tab characters displayed inside a row?

**Expected:** They are expanded into spaces up to the next eight-column tab stop so cursor positioning and highlighting line up with the visible output.

- **shine / baseline** (score 4): Tab characters are displayed as spaces inside a row.
- **shine / detail**   (score 4): Tab characters are displayed as spaces inside a row.
- **shine / adapted**  (score 4): Tab characters are displayed as spaces inside a row.

#### `rows_editing_persistence_1_0004`

**Q:** What does `editorUpdateRow` do?

**Expected:** When a row's underlying characters change, it rebuilds the rendered string and recomputes the highlight bytes for that row.

- **shine / baseline** (score 2): `editorUpdateRow` updates the visible row and scroll offset to the same one.
- **shine / detail**   (score 4): `editorUpdateRow` mirrors the original row of the row, creates a buffer with syntax highlighting, and a parallel buffer for the rendered view.
- **shine / adapted**  (score 4): `editorUpdateRow` rebuilds the visible rows, strips trailing newlines, recomputes the highlight bytes, and marks the buffer dirty.

#### `rows_editing_persistence_1_0005`

**Q:** What happens when a new row is inserted through `editorInsertRow`?

**Expected:** The row array is reallocated, later rows shift down with their indices fixed, the new row's buffers are initialized, the rendered view is recomputed, and the file is marked dirty.

- **shine / baseline** (score 1): A new row is inserted through `editorInsertRow`.
- **shine / detail**   (score 3): When a new row is inserted through `editorInsertRow`, it creates a fresh row in the editor, initializes the row's underlying data structures, and inserts a parallel rendering version used for display, syntax highlighting, and line-column tracking.
- **shine / adapted**  (score 2): When a new row is inserted through `editorInsertRow`, it is appended to the original array of `Row` structures.

#### `rows_editing_persistence_1_0006`

**Q:** What does `editorFreeRow` release before a row is deleted?

**Expected:** The row's heap allocations for the original text, the rendered buffer, and the highlight bytes.

- **shine / baseline** (score 1): `editorFreeRow` releases the previous row.
- **shine / detail**   (score 1): `editorFreeRow` releases the previous row
- **shine / adapted**  (score 2): `editorFreeRow` releases the previous row's buffer

#### `rows_editing_persistence_1_0007`

**Q:** What happens if `editorRowInsertChar` is asked to insert a character past the current end of a line?

**Expected:** It first pads the row with spaces up to the requested column, then writes the character.

- **shine / baseline** (score 1): The editor inserts a newline and scrolls the current buffer down to the new line.
- **shine / detail**   (score 2): If `editorRowInsertChar` is asked to insert a character past the current end of a line, it does nothing.
- **shine / adapted**  (score 1): If `editorRowInsertChar` is asked to insert a character past the current end of a line, it does nothing.

#### `rows_editing_persistence_1_0008`

**Q:** How does `editorInsertChar` translate the visible cursor into a file position?

**Expected:** It translates the visible cursor and scroll offsets into a file row and column, and creates an empty row at the end of the file when typing past the last line.

- **shine / baseline** (score 2): EditorInsertChar translates the visible cursor into a file position by expanding the previous characters and the highlight into their display form, then advancing the cursor in the rendered view by one block, and recomputing the highlight bytes.
- **shine / detail**   (score 1): `editorInsertChar` translates the visible cursor into a file position by expanding the previous characters and the highlight later.
- **shine / adapted**  (score 2): `editorInsertChar` translates the visible cursor into a file position by expanding the previous characters and the highlight later, then advancing the file cursor to the next line.

#### `rows_editing_persistence_1_0009`

**Q:** What does `editorInsertNewline` do when the cursor is in the middle of a row?

**Expected:** It splits the row at the cursor: the right half becomes a new row beneath the original, the original is truncated and its render data refreshed, and the cursor moves to the start of the new line.

- **shine / baseline** (score 2): It splits the current row into two, creates a fresh row for the original text, and joins the previous row and the new one with their respective trailing comment lines.
- **shine / detail**   (score 4): `editorInsertNewline` splits the current row into two rows, the first row containing the original characters up to the cursor, and the second row containing the original characters and the cursor's shift.
- **shine / adapted**  (score 5): `editorInsertNewline` splits the current row into two rows, the first row containing the original characters up to the cursor, and the second row containing the original characters and the cursor later shifted down to the new row.

#### `rows_editing_persistence_1_0010`

**Q:** How does `editorDelChar` behave when the cursor is at column zero?

**Expected:** It joins the current row to the previous one by appending the text and then deleting the original.

- **shine / baseline** (score 2): `editorDelChar` does nothing when the cursor is at column zero.
- **shine / detail**   (score 2): `editorDelChar` does nothing when the cursor is at column zero.
- **shine / adapted**  (score 2): `editorDelChar` tries to shift the current line to the left, truncating it to the exact new length.

#### `rows_editing_persistence_1_0011`

**Q:** What does `editorOpen` do when the requested file does not exist?

**Expected:** It returns success, leaves the buffer empty, and lets the user save under that name later.

- **shine / baseline** (score 2): `editorOpen` tries to open the file, creates a fresh file if it does not exist, and marks the file in the dirty flag.
- **shine / detail**   (score 1): `editorOpen` tries to open the file through `erow` and `highlight` if the requested file exists, or tries to open it in the editor through `erow` and `highlight` if it does not.
- **shine / adapted**  (score 2): `editorOpen` tries to open the file through `erow` and `cursor`

#### `rows_editing_persistence_1_0012`

**Q:** How does `editorRowsToString` serialize the in-memory rows for saving?

**Expected:** It concatenates all rows with a newline between each.

- **shine / baseline** (score 4): `editorRowsToString` serializes the in-memory rows by concatenating each row's text with a newline, stripping trailing newlines, and joining the resulting strings.
- **shine / detail**   (score 2): `editorRowsToString` serializes the in-memory rows by concatenating the original characters of each line, a parallel rendered version used for display, syntax highlighting, and line end, and a small flag tracking whether the previous line ended with a newline.
- **shine / adapted**  (score 1): `editorRowsToString` serializes the in-memory rows by concatenating the original characters of each line, a parallel array of metadata strings, the highlight flag, and the buffer's underlying data.

#### `rows_editing_persistence_1_0013`

**Q:** What does `editorSave` do before reporting success?

**Expected:** It opens or creates the target file, truncates it to the exact new length, writes the buffer in one call, closes the file, clears the dirty flag, and reports the number of bytes written.

- **shine / baseline** (score 1): `editorSave` reports success after appending the current buffer to the target file.
- **shine / detail**   (score 1): `editorSave` clears the dirty flag, records the previous buffer version, and marks the current buffer as dirty.
- **shine / adapted**  (score 1): `editorSave` clears the previous buffer and rebuilds it to the exact characters of the original, then marks the buffer as refreshed.

</details>
