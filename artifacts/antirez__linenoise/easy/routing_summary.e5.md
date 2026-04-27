# E5 Embedding Routing Summary for antirez/linenoise

## Setup

Router: embedding cosine similarity  
Embedding model: `intfloat/e5-small-v2`  
Repository: `antirez/linenoise`  
Candidate documents:
- `overview_purpose_1`
- `raw_terminal_input_1`
- `history_completion_hints_1`
- `refresh_editing_1`

Each candidate routing text is derived from its design document and QA examples. Each QA row provides a gold `document_id`, used as the correct route.

## Results

| QA set | n | Top-1 accuracy | Top-2 accuracy | Top-3 accuracy | Top-4 accuracy |
|---|---:|---:|---:|---:|---:|
| `history_completion_hints_1` | 13 | 1.000 | 1.000 | 1.000 | 1.000 |
| `overview_purpose_1` | 12 | 0.667 | 0.750 | 0.917 | 1.000 |
| `raw_terminal_input_1` | 13 | 0.769 | 0.923 | 1.000 | 1.000 |
| `refresh_editing_1` | 13 | 0.615 | 1.000 | 1.000 | 1.000 |
| **Overall** | **51** | **0.765** | **0.922** | **0.980** | **1.000** |

## Interpretation

The E5 embedding router achieves solid top-1 accuracy and very strong top-k recall. Top-2 routing recovers most missed top-1 routes, and top-3 includes the correct document for 50 of 51 questions. The weakest categories are overview and refresh/editing, likely because their concepts overlap with other linenoise topics. This supports evaluating top-k routing for downstream LoRA selection or composition, while keeping answer-quality evaluation separate from routing accuracy.
