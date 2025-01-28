import torch
import einops
import faiss
from .topk_attn import LlamaTopkAttention
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer


def convert_cache_to_topk(past_key_values):
    """
    Takes in a cache and returns a cache that can be used for topk decoding
    """
    raise NotImplementedError


def convert_model_to_topk(model):
    """
    Takes in a model and returns a model that can use topk decoding
    """
    for layer in model.model.layers:
        assert isinstance(layer, LlamaDecoderLayer)
        config = layer.self_attn.config
        layer.self_attn = LlamaTopkAttention(config, layer_idx=layer.layer_idx)

    return model
