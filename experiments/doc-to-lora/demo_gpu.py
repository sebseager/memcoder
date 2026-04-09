import os
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "vendor" / "doc-to-lora" / "src"),
)

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_D2L_ROOT = PROJECT_ROOT / "vendor" / "doc-to-lora"


@contextmanager
def pushd(path: Path):
    """Temporarily change cwd so doc-to-lora can resolve relative assets."""
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


checkpoint_path = "trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin"
state_dict = torch.load(checkpoint_path, weights_only=False)
state_dict["ctx_encoder_args"].quantize_ctx_encoder = False

model = ModulatedPretrainedModel.from_state_dict(
    state_dict,
    train=False,
    use_sequence_packing=False,
    use_flash_attn=True,
)
model.reset()
with pushd(VENDOR_D2L_ROOT):
    tokenizer = get_tokenizer(model.base_model.name_or_path)

data_dir = VENDOR_D2L_ROOT / "data"
doc = open(data_dir / "sakana_wiki.txt", "r").read()

with pushd(VENDOR_D2L_ROOT):
    model.internalize(doc)

chat = [{"role": "user", "content": "Tell me about Sakana AI."}]
chat_ids = tokenizer.apply_chat_template(
    chat,
    add_special_tokens=False,
    return_attention_mask=False,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

output = model.generate(input_ids=chat_ids, max_new_tokens=200)
print(tokenizer.decode(output[0], skip_special_tokens=True))
