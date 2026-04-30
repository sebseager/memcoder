# pytest-dev__pytest easy ledger

Source commit: `5dfd4eab7d328266fb3788dcbb031c9d67903daf`

## Overview / Purpose

- [overview_purpose_1] Overview / Purpose (easy) - Explains pytest's purpose as a Python testing framework, its public entry points, documentation shape, and the high-level path from CLI invocation through configuration, session setup, collection, and test execution. Example questions: What is pytest's overall purpose as described by its README and package docstring?; How does pytest support both small test files and larger functional test suites?; What kinds of documentation sections does pytest provide for users and plugin authors?; How does pytest turn configured command-line arguments into collection and test execution?; Why is pytest's plugin-aware session engine central to the framework's design?

## Setup and Development

- [setup_development_1] Setup and Development (easy) - Explains how pytest is installed, how contributors prepare a local checkout, and how tox, pre-commit, documentation builds, changelog entries, and release tooling fit into the development workflow. Example questions: What is the documented way to verify a user-level pytest installation from the command line?; How does pytest use pre-commit in its contributor workflow?; What does the pytest tox docs environment build, and where does it put the HTML output?; When can a pytest contributor skip adding a changelog entry?; How is pytest's release preparation split between the automated GitHub workflow and the manual tox-based path?

## Public API Interface

- [public_api_interface_1] Public API Interface (easy) - Explains how pytest's top-level public API is assembled, where the reference documentation points users, and how key exported helpers for comparisons, exceptions, outcomes, and warnings behave. Example questions: How does pytest expose internal helpers through the public pytest namespace?; What public pytest helpers raise dedicated outcome exceptions for fail, skip, xfail, and exit?; How does pytest.warns record, validate, and re-emit warnings from a test block?; What is the role of the recwarn fixture in pytest's public warning API?; How does pytest's public exception-group assertion helper differ from a simple containment check?

## Configuration, Plugins, and Hooks

- [config_plugins_hooks_1] Configuration, Plugins, and Hooks (easy) - Explains how pytest builds configuration, discovers and controls plugins, loads conftest files, and exposes hook-based extension points during startup and test execution. Example questions: How does pytest discover built-in, entry point, environment, and conftest plugins during startup?; What role does PytestPluginManager play in registering plugins and loading hook specifications?; How are local conftest.py files scoped and cached for directory-specific hook behavior?; What does pytest's hook argument pruning allow older plugins to do safely?; How can pytest disable automatic third-party plugin autoloading while still allowing explicit plugins?

## Collection and Node Model

- [collection_nodes_1] Collection and Node Model (easy) - Explains how pytest turns command-line inputs into a collection tree, how collection nodes are constructed and identified, and how filesystem and Python collectors produce runnable test items. Example questions: How does pytest build its collection tree from directories and Python test files?; What information does a pytest node store for reporting and selection?; How do pytest's import modes affect collection of Python test modules?; When does pytest create Module, Class, and Function nodes during Python collection?; How do ignore rules and deselection fit into pytest's collection phase?

## Fixtures, Parametrization, and Marks

- [fixtures_parametrize_marks_1] Fixtures, Parametrization, and Marks (easy) - Explains how pytest turns fixture parameters and parametrize marks into collected test cases, how per-parameter marks are stored on items, and how skip, xfail, and marker selection consume those marks. Example questions: How do fixture params flow through Metafunc during pytest collection?; What happens to marks attached with pytest.param in a parametrized test or fixture?; How does pytest build item names and marker state for parametrized Function items?; Why are marks rejected when someone applies them directly to fixture functions?; How do skip and xfail evaluation use marks that originated from parameter sets?

## Assertions, Warnings, and Outcomes

- [assertions_warnings_outcomes_1] Assertions, Warnings, and Outcomes (easy) - Explains how pytest turns plain asserts into useful failure reports, how warning capture and warning assertions work, and how skip, fail, xfail, exit, raises, and import-based outcomes are represented. Example questions: How does pytest produce detailed failure output from normal Python assert statements?; What assertion comparison details does pytest add for strings, sequences, sets, and dictionaries?; How are warning filters from configuration, command-line options, and marks applied during a pytest run?; When should a test use pytest.importorskip instead of a skip marker?; How do pytest's fail, skip, xfail, and exit helpers communicate outcomes internally?

## Runtime, Reporting, and IO Helpers

- [runtime_reporting_io_1] Runtime, Reporting, and IO Helpers (easy) - Explains how pytest runs test phases, turns phase results into reports, attaches captured output and logs, prints terminal progress and summaries, and manages tmp_path directories. Example questions: How does pytest update PYTEST_CURRENT_TEST while a test item is running?; What information do pytest report objects expose for captured output and long failure text?; How does TerminalReporter choose between compact progress letters and verbose per-test output?; Where does pytest collect timing data for the slowest durations summary?; How does tmp_path_factory choose and manage the base temporary directory?
