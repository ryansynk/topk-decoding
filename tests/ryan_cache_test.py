import argparse
import torch
import einops
import os
from transformers import AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--hf", action='store_true')
    return parser.parse_args()

def get_hf_cache_as_tens(cache):
    tens = []
    for l in range(len(cache)):
        tens_list = []
        for t in cache[l]:
            tens_list.append(t.cpu())
        tens.append(torch.cat(tens_list, dim=0))
        
    tens = torch.stack(tens, dim=0)
    tens = einops.rearrange(tens, "l kv h n d -> kv l h n d")
    return tens.cpu()

def get_hf_cache(x, dtype, device):
    model = AutoModelForCausalLM.from_pretrained(
        "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        attn_implementation="sdpa",
        torch_dtype=dtype,
        device_map=device
    )
    model = model.eval()
    with torch.no_grad():
        outputs = model(x.to(model.device), use_cache=True)
    return get_hf_cache_as_tens(outputs.past_key_values)

def get_hf_forward(x, position_ids, dtype):
    model = AutoModelForCausalLM.from_pretrained(
        "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        attn_implementation="sdpa",
        torch_dtype=dtype,
        device_map="auto"
    )
    model = model.eval()
    #N = x.shape[-2]
    #dtype, device = model.dtype, model.device
    #min_dtype = torch.finfo(dtype).min
    #attn_mask = torch.full(
    #    (N, N),
    #    fill_value=min_dtype,
    #    dtype=dtype,
    #    device=device
    #)
    #attn_mask = torch.triu(attn_mask, diagonal=1).expand(32, -1, -1).unsqueeze(0)
    with torch.no_grad():
        fwd = model.model.layers[0].self_attn.forward
        output = fwd(hidden_states=x.to(model.device), position_ids=position_ids.to(model.device), attention_mask=attn_mask)
    return output[0]

def get_my_cache(x, dtype, device):
    model = AutoModelForCausalLM.from_pretrained(
        "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
        attn_implementation="sdpa",
        torch_dtype=dtype,
        device_map=device
    )
    model = model.eval()
    with torch.no_grad():
        outputs = model(x.to(model.device), use_cache=True, num_mlp_splits=4, cpu_cache=True)
    return get_hf_cache_as_tens(outputs.past_key_values)

def get_my_forward(x, position_ids, dtype):
    #model = AutoModelForCausalLM.from_pretrained(
    #    "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
    #    attn_implementation="sdpa",
    #    torch_dtype=dtype,
    #    device_map="auto"
    #)
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        attn_implementation="sdpa",
        torch_dtype=dtype,
        device_map="auto"
    )
    model = model.eval()
    with torch.no_grad():
        fwd = model.model.layers[0].self_attn.forward
        output = fwd(hidden_states=x.to(model.device), position_ids=position_ids.to(model.device), num_mlp_splits=4)
    return output[0]

def main():
    args = parse_args()
    torch.manual_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    #torch.set_float32_matmul_precision("highest")
    dtype = torch.bfloat16
    N = 4096
    D = 4096
    #num_layers = 0
    #x = torch.rand((1, N, D), dtype=dtype)
    #position_ids = torch.arange(N).unsqueeze(0)
    #hf_out = get_hf_forward(x, position_ids, dtype)
    #my_out = get_my_forward(x, position_ids, dtype)
    #torch.testing.assert_close(hf_out, my_out)

    x_int = torch.randint(low=0, high=1000, size=(1, N)).cuda()
    if args.hf:
        cache = get_hf_cache(x_int, dtype, args.device)
        fname = f"hf_cache_N_{N}.pt"
    else:
        cache = get_my_cache(x_int, dtype, args.device)
        fname = f"low_memory_cache_N_{N}.pt"

    torch.save(cache, fname)
    

if __name__=="__main__":
    main()
