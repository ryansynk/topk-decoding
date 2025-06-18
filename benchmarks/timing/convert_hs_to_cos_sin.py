import argparse
import re
import os
import torch
import glob
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

"""
This script converts hs to pos_embs, needed for streamingLLM
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_directory", type=str, default="/fs/nexus-projects/FAST_Attention/ann/"
    )
    parser.add_argument("--task", type=str, default="niah_single_1_plot")
    parser.add_argument("--example", type=int, default=0)
    N_values = [4096, 8192, 16384, 32768, 65536, 131072]
    parser.add_argument(
        "--N",
        required=True,
        type=str,
        choices=[str(N) for N in N_values] + ["all"],
        help="Size of context length",
    )
    args = parser.parse_args()
    if args.N == "all":
        args.N = [int(n) for n in N_values]
    else:
        args.N = [int(args.N)]
    return args


def get_cos_sin_from_hs(hs, model, cache_position):
    cache_position_ids = torch.arange(0, cache_position).unsqueeze(0).to(model.device)
    pos_embeddings = model.model.rotary_emb(hs[0], cache_position_ids)
    cos, sin = pos_embeddings
    cos = cos.cpu()
    sin = sin.cpu()
    return torch.stack((cos, sin), dim=0)


def get_cache_position(cache_path):
    cache = torch.load(cache_path)
    return cache.shape[-2]


def get_base_path(args, model_name, N):
    base_path = args.data_directory
    base_path = os.path.join(base_path, model_name.split("/")[-1])
    base_path = os.path.join(base_path, str(N))
    base_path = os.path.join(base_path, args.task)
    base_path = os.path.join(base_path, f"example_{args.example}")
    return base_path


def main():
    args = get_args()
    model = AutoModelForCausalLM.from_pretrained(
        "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    for n in args.N:
        base_path = get_base_path(
            args, "gradientai/Llama-3-8B-Instruct-Gradient-1048k", n
        )
        cache_path = os.path.join(base_path, "kv_cache.pt")
        cache_pos = get_cache_position(cache_path)
        hidden_states_path = os.path.join(base_path, "hs.pt")
        hidden_states = torch.load(hidden_states_path)
        pos_embs = get_cos_sin_from_hs(hidden_states, model, cache_pos)
        pos_emb_path = os.path.join(base_path, "pos_embs.pt")
        torch.save(pos_embs, pos_emb_path)


if __name__ == "__main__":
    main()
