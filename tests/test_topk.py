import pytest
import torch
import topk_decoding
from topk_decoding import topk_decoding
from transformers import AutoModelForCausalLM


def test_import_convert_cache_to_topk():
    with pytest.raises(NotImplementedError) as e_info:
        topk_decoding.convert_cache_to_topk(None)


def test_import_convert_model_to_topk():
    model = AutoModelForCausalLM.from_pretrained(
        "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = topk_decoding.convert_model_to_topk(model)
    for layer in model.model.layers:
        assert isinstance(layer.self_attn, topk_decoding.LlamaTopkAttention), type(
            layer.self_attn
        )
