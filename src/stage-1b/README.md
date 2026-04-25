# Stage 1b — IFT SHINE on Code

This folder contains the Stage 1b workflow for building code IFT triples, training SHINE with the upstream `ift-c1qa` recipe, running held-out inference, and doing a lightweight non-Docker evaluation.

## Outputs

- `data/ift_c1qa_code_train.json`: SHINE-compatible training list with `context`, `conversations`, `contextlen`, and `conversationlen`.
- `outputs/train_triples.records.jsonl`: auditable source metadata for each training triple.
- `outputs/heldout_instances.jsonl`: 20 repo-disjoint held-out triples for Stage 1b inference.
- `checkpoints/<NAME>/train`: Stage 1b checkpoints. The launcher symlinks the Stage 1a `iftpwc` checkpoint as input; it does not copy or edit it.

## Build Triples

```bash
cd src
MODE=build stage-1b/scripts/run_stage1b.sh --force
```

Useful knobs:

```bash
MODE=build MAX_TRIPLES=1500 HELDOUT=20 stage-1b/scripts/run_stage1b.sh --force
```

The builder filters SWE-rebench training rows to be repo-disjoint from the Stage 0/1 held-out set, applies the same basic function filters as Stage 0, and uses the same Stage 1a slicing structure.

## Train

```bash
cd src
MODE=train NUM_GPUS=1 stage-1b/scripts/run_stage1b.sh
```

Defaults are one epoch and `lr=1e-5`. The script creates a writable SHINE overlay at `stage-1b/shine_work`, points `data/ift_c1qa.json` to the Stage 1b training JSON, points the required `iftpwc` checkpoint to Stage 1a, and writes new checkpoints under Stage 1b.

## Inference

```bash
cd src
MODE=infer stage-1b/scripts/run_stage1b.sh --force
```

By default this uses the latest `checkpoint-*` under `stage-1b/checkpoints/$NAME/train` and writes `outputs/stage1b_predictions.jsonl`.

## Lightweight Evaluation

```bash
cd src
MODE=eval stage-1b/scripts/run_stage1b.sh --baseline-b 0.0 --ceiling-c 0.2
```

This computes exact match, token F1, BLEU-4, and AST validity without the SWE-rebench Docker harness. It also reports a proxy recovery ratio:

```text
rho = (E - B) / (C - B)
```

Use pass@1 values for `B` and `C` when available. The lightweight score is for iteration on HPC outputs; the final paper number should still use the Docker harness.
