import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parents[2] / "vendor" / "doc-to-lora" / "src"))

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

# Monkey-patch model loading so CPU runs do not try to use FlashAttention2
_original_from_pretrained = AutoModelForCausalLM.from_pretrained

def _cpu_safe_from_pretrained(*args, **kwargs):
    kwargs["attn_implementation"] = "eager"
    return _original_from_pretrained(*args, **kwargs)

AutoModelForCausalLM.from_pretrained = _cpu_safe_from_pretrained

# Load the hypernetwork + base model
checkpoint_path = "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"
state_dict = torch.load(
    checkpoint_path,
    map_location=torch.device("cpu"),
    weights_only=False,
)
model = ModulatedPretrainedModel.from_state_dict(
    state_dict,
    train=False,
    use_sequence_packing=False,
)
model.reset()
tokenizer = get_tokenizer(model.base_model.name_or_path)

doc = open("data/sakana_wiki.txt", "r").read()

# Internalize the document into the model's hypernetwork
model.internalize(doc)

# Now query without the document in context
chat = [{"role": "user", "content": "Tell me about Sakana AI."}]
chat_ids = tokenizer.apply_chat_template(
    chat,
    add_special_tokens=False,
    return_attention_mask=False,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

output = model.generate(chat_ids, max_new_tokens=200)
print(tokenizer.decode(output[0], skip_special_tokens=True))
