"""Registry of models you can run.

Each loader takes a single `use_topk` flag and returns `(model, tokenizer)`.
- `use_topk=False` → plain HF model, regular attention.
- `use_topk=True`  → same model with top-k attention patched in.

To add a new model: write a loader function and add it to `MODELS`.
"""

import types
from typing import Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from topk_decoding import AutoTopkModelForCausalLM
from topk_decoding.monkey_patch import _topk_generate, apply_topk_to_llama


ModelAndTokenizer = Tuple[PreTrainedModel, PreTrainedTokenizerBase]


def _patch_topk_in_place(model: LlamaForCausalLM) -> LlamaForCausalLM:
    """Apply top-k attention to a model that was built from scratch (not via from_pretrained).

    AutoTopkModelForCausalLM does this work as part of from_pretrained, but for the
    synthetic tiny model we already have a LlamaForCausalLM and just need the same
    in-place patching the wrapper would do.
    """
    apply_topk_to_llama(model, attn=True, mlp=False)
    model.original_generate = model.generate
    model.generate = types.MethodType(_topk_generate, model)
    return model


class _ByteTokenizer:
    """Trivial byte-level tokenizer for the tiny synthetic LLaMA.

    Maps each input byte to its integer value (0-255). Matches `vocab_size=256` in
    `load_tiny_llama`. Not a real tokenizer; just enough to feed strings into the model.
    """

    def __call__(self, text: str, return_tensors: str = "pt"):
        ids = torch.tensor([[b for b in text.encode("utf-8")]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return bytes(int(x) & 0xFF for x in ids).decode("utf-8", errors="replace")


def load_tiny_llama(use_topk: bool) -> ModelAndTokenizer:
    """Random-init tiny LLaMA. CPU-friendly, no download, ~1M params total."""
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=2048,
    )
    model = LlamaForCausalLM(config).eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    if use_topk:
        model = _patch_topk_in_place(model)
    return model, _ByteTokenizer()


def load_llama_8b_1048k(use_topk: bool) -> ModelAndTokenizer:
    """The production model from the existing test. Requires GPU and ~16GB download."""
    name = "gradientai/Llama-3-8B-Instruct-1048k"
    dtype = torch.bfloat16
    cls = AutoTopkModelForCausalLM if use_topk else AutoModelForCausalLM
    model = cls.from_pretrained(name, torch_dtype=dtype).to("cuda").eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer


MODELS = {
    "tiny-llama": load_tiny_llama,
    "llama-3-8b-1048k": load_llama_8b_1048k,
}
