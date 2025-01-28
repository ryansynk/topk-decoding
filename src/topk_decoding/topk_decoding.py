import torch
import einops
import faiss
from transformers import AutoModelForCausalLM

def convert_cache_to_topk(past_key_values):
    """
    Takes in a cache and returns a cache that can be used for topk decoding
    """
    raise NotImplementedError

def convert_model_to_topk(model):
    """
    Takes in a model and returns a model that can use topk decoding
    """
    raise NotImplementedError
