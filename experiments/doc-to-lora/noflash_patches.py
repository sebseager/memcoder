"""Monkey-patches to run doc-to-lora models without the flash_attn package."""

import torch
import torch.nn.functional as F
import transformers.modeling_utils as modeling_utils


def apply_noflash_patches():
    """Patch the idefics2 perceiver to work without flash_attn.

    Must be called BEFORE importing/instantiating any model classes that use
    the perceiver (i.e. before ModulatedPretrainedModel.from_state_dict).
    """
    # 1. No-op _autoset_attn_implementation so transformers doesn't validate
    #    that flash_attn is installed. The perceiver configs are created with
    #    attn_implementation="flash_attention_2" hardcoded in vendor code;
    #    we keep that value but replace the actual attention class below.
    @classmethod
    def _noop_autoset_attn(cls, config, *args, **kwargs):
        return config

    modeling_utils.PreTrainedModel._autoset_attn_implementation = _noop_autoset_attn

    # 2. Import idefics2 module (safe – the flash_attn imports are guarded)
    from ctx_to_lora.modeling import idefics2
    from ctx_to_lora.modeling.idefics2 import Idefics2PerceiverAttention, repeat_kv

    # 3. Provide an unpad_input shim (used by Idefics2PerceiverResampler.forward
    #    when attention_mask is not None).
    def _unpad_input(hidden_states, attention_mask):
        seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen = seqlens.max().item()
        cu_seqlens = F.pad(
            torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
        )
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        return flat[indices], indices, cu_seqlens, max_seqlen, seqlens

    idefics2.unpad_input = _unpad_input

    # 4. SDPA-based attention with the same forward signature as
    #    Idefics2PerceiverFlashAttention2 (accepts **kwargs like cu_seq_lens
    #    and silently ignores them).
    class _SDPPerceiverAttention(Idefics2PerceiverAttention):
        def forward(
            self,
            latents,
            is_cross_attn,
            context=None,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            **kwargs,
        ):
            bsz, q_len, _ = latents.size()
            kv_inp = context if is_cross_attn else latents

            query_states = self.q_proj(latents).view(
                *latents.shape[:2], self.num_heads, self.head_dim
            ).transpose(1, 2)
            key_states = self.k_proj(kv_inp).view(
                *kv_inp.shape[:2], self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            value_states = self.v_proj(kv_inp).view(
                *kv_inp.shape[:2], self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=None, dropout_p=0.0, is_causal=False,
            )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(
                bsz, q_len, self.num_heads * self.head_dim
            )
            attn_output = self.o_proj(attn_output)

            return attn_output, None, (key_states, value_states) if use_cache else None

    # 5. Swap the class in the registry (keyed by "flash_attention_2")
    idefics2.IDEFICS2_PERCEIVER_ATTENTION_CLASSES["flash_attention_2"] = (
        _SDPPerceiverAttention
    )
