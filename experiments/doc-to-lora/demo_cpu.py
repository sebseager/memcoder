import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[2] / "vendor" / "doc-to-lora" / "src"))

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

# Load the hypernetwork + base model
checkpoint_path = "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"
state_dict = torch.load(checkpoint_path, weights_only=False)
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
