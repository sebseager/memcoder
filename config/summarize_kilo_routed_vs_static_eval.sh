#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python - <<'PY'
import json
from pathlib import Path
from collections import defaultdict

path = Path("artifacts/antirez__kilo/easy/lora_routed_vs_static_results.jsonl")
rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

by_cond = defaultdict(list)
by_doc_cond = defaultdict(list)

for r in rows:
    by_cond[r["condition"]].append(r)
    by_doc_cond[(r["document_id"], r["condition"])].append(r)

print("Overall:")
for cond, items in sorted(by_cond.items()):
    n = len(items)
    avg_f1 = sum(x["scores"]["token_f1"] for x in items) / n
    exact_contains = sum(bool(x["scores"].get("exact_or_contains")) for x in items) / n
    print(f"{cond}: n={n}, avg_token_f1={avg_f1:.3f}, exact_or_contains={exact_contains:.3f}")

print("\nBy document:")
for (doc, cond), items in sorted(by_doc_cond.items()):
    n = len(items)
    avg_f1 = sum(x["scores"]["token_f1"] for x in items) / n
    print(f"{doc:32s} {cond:40s} avg_token_f1={avg_f1:.3f}")

by_qa = defaultdict(dict)
for r in rows:
    by_qa[r["qa_id"]][r["condition"]] = r

routed_beats_best_static = 0
best_static_beats_routed = 0
ties = 0
deltas = []

for qa_id, runs in by_qa.items():
    routed = runs.get("routed_top1")
    static_runs = [r for c, r in runs.items() if c.startswith("static:")]
    if not routed or not static_runs:
        continue

    best_static = max(static_runs, key=lambda r: r["scores"]["token_f1"])
    delta = routed["scores"]["token_f1"] - best_static["scores"]["token_f1"]
    deltas.append(delta)

    if delta > 1e-9:
        routed_beats_best_static += 1
    elif delta < -1e-9:
        best_static_beats_routed += 1
    else:
        ties += 1

print("\nRouted vs best static per question:")
print(f"routed > best_static: {routed_beats_best_static}")
print(f"routed < best_static: {best_static_beats_routed}")
print(f"routed = best_static: {ties}")
if deltas:
    print(f"avg routed_minus_best_static token_f1: {sum(deltas)/len(deltas):.3f}")
PY
