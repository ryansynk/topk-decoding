import torch
import einops
import faiss
from .topk_attn import TopkAttention
from .topk_cache import TopkCache
from transformers import AutoModelForCausalLM, DynamicCache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


def convert_cache_to_topk(past_key_values):
    """
    Takes in a cache and returns a cache that can be used for topk decoding
    """
    if isinstance(past_key_values, DynamicCache):
        topk_key_values = TopkCache.from_dynamic_cache(past_key_values)
    else:
        raise NotImplementedError

    return topk_key_values


def convert_model_to_topk(model, topk_k):
    """
    Takes in a model and returns a model that can use topk decoding
    """
    device = model.device
    dtype = model.dtype
    for layer_idx, layer in enumerate(model.model.layers):
        assert isinstance(layer, LlamaDecoderLayer)
        config = layer.self_attn.config
        layer.self_attn = TopkAttention(config, layer_idx=layer_idx)
        layer.self_attn.topk_k = topk_k

    return model.to(dtype).to(device)
