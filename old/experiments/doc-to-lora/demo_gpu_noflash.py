import sys
from pathlib import Path

import torch
import transformers.modeling_utils as modeling_utils
from checkpoint_config import resolve_checkpoint_path

sys.path.insert(0, str(Path(__file__).parents[2] / "vendor" / "doc-to-lora" / "src"))

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel


@classmethod
def _force_eager_attn(cls, config, *args, **kwargs):
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "eager"
    return config


modeling_utils.PreTrainedModel._autoset_attn_implementation = _force_eager_attn

checkpoint_path = resolve_checkpoint_path()
print(f"Loading checkpoint: {checkpoint_path}")
state_dict = torch.load(checkpoint_path, weights_only=False)

from ctx_to_lora.modeling import idefics2

# monkey-patch the missing key
available = next(iter(idefics2.IDEFICS2_PERCEIVER_ATTENTION_CLASSES.values()))
idefics2.IDEFICS2_PERCEIVER_ATTENTION_CLASSES["eager"] = available

model = ModulatedPretrainedModel.from_state_dict(
    state_dict,
    train=False,
    use_sequence_packing=False,
)
model.reset()
tokenizer = get_tokenizer(model.base_model.name_or_path)

doc = (Path(__file__).resolve().parent / "data" / "sakana_wiki.txt").read_text()
model.internalize(doc)

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
