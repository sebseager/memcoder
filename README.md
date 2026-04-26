## Ideas for what to turn into lora


1. Bug fix trajectories
    Have some representation of the model trying and succeeding to solve various 
    challenges. This teaches it how to format tool calls correctly, patterns that 
    tend to work, etc. You have the regular model (or maybe a stronger teacher model) 
    generate trajectories of tool calls. Then put those into the hypernetwork and 
    do weight updates based on that. That way the model "learns" what trajectories 
    or patterns tend to work better.
    - Negative "award" for stuff that didn't work

2. Static chunks (easy)
    - Take with some granularity chunks from the repo, toss them into HN, get LoRAs
    - Store a mapping between code areas and their associated LoRA

3. Dynamic chunks (slightly harder)
    - Have agent explore chunk
    - Agent then writes a design document on the chunk, how it works, its conventions, and how it relates to other chunks
    - This design doc then gets turned into a lora

4. 
    
## TODO

1. What goes in the input to the LORA?
2. How is the output managed and accessed?

## Repository layout

- `vendor/`: third-party implementation dependencies used by this project.
- `target_repos/`: third-party repositories used as inputs for design-document
  and QA generation experiments.

Target repositories should be added as git submodules using the path convention:

```text
target_repos/{owner}__{repo}
```

For example:

```sh
git submodule add https://github.com/antirez/kilo target_repos/antirez__kilo
```

## Running SHINE eval

The initial evaluation script is configured by `config/shine_eval_demo.yaml`,
which starts from the settings in `vendor/SHINE/inference.ipynb`.

```sh
python scripts/run_shine_eval.py --config config/shine_eval_demo.yaml
```

Cluster-specific paths can be edited in the config or overridden on the command
line:

```sh
python scripts/run_shine_eval.py \
  --config config/shine_eval_demo.yaml \
  --model-path /path/to/Qwen3-8B \
  --checkpoint-dir /path/to/checkpoint-epoch-2
```
