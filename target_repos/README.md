# Target Repositories

This directory holds third-party repositories used as inputs for design-document
and QA generation experiments.

Use this directory for repositories being studied or evaluated by MemCoder. Keep
implementation dependencies in `vendor/`.

Recommended submodule naming:

```text
target_repos/{owner}__{repo}
```

For example:

```text
target_repos/antirez__kilo
```

Add a new target repository with:

```sh
git submodule add https://github.com/{owner}/{repo} target_repos/{owner}__{repo}
```
