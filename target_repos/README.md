# Target Repositories

This directory holds third-party repositories used as inputs for design-document
and QA generation experiments.

Use this directory for repositories being studied or evaluated by MemCoder. Keep
implementation dependencies in `vendor/`.

Submodule naming:

```text
target_repos/{owner}__{repo}
```

For example:

```text
target_repos/antirez__kilo
```

Current target repository submodules:

- `antirez__kilo`: small C text editor used in the strongest pilot signal.
- `marimo-team__marimo`: Python reactive notebook system used in the broader
  easy-tier pilot.
- `fogleman__Craft`: candidate target repo, checked out for future expansion.
- `psf__requests`: candidate Python target repo for future expansion.
- `pytest-dev__pytest`: candidate Python target repo for future expansion.

Add a new target repository with:

```sh
git submodule add https://github.com/{owner}/{repo} target_repos/{owner}__{repo}
```
