import os
import argparse
import time
import csv
import torch
import datasets
from datetime import datetime
from transformers import AutoModelForCausalLM, DynamicCache, AutoTokenizer
from topk_decoding import AutoTopkModelForCausalLM, TopkCache

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gradientai/Llama-3-8B-Instruct-Gradient-1048k", type=str) 
    parser.add_argument("--dataset_dir", type=str, default="/fs/nexus-projects/FAST_Attention/ann", help="Path to dataset json files")
    parser.add_argument("--decode_strategy",type=str, choices=["full", "offload", "topk_flat", "topk_ivf", "streaming_llm"], required=True)
    parser.add_argument("--nondeterministic", action="store_true", help="Turn on sampling during generation (output will be deterministic).")
    parser.add_argument("--max_new_tokens", default=5, type=int)
    parser.add_argument("--N", type=int, choices=[4096, 8192, 16384, 32768, 65536, 131072], required=True)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()

def get_kwargs(args, cache):
    if args.nondeterministic:
        kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            top_p=1,
            temperature=0.7,
            num_return_sequences=1,
        )
    else:
        kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=1,
            do_sample=False,
            num_beams=1,
        )
    if args.decode_strategy in ["topk_flat", "topk_ivf"]:
        kwargs['k'] = args.k
    kwargs['past_key_values'] = cache
    return kwargs

def get_cache_path(args):
    path = os.path.join(args.dataset_dir, args.model.split("/")[-1])
    path = os.path.join(path, str(args.N))
    path = os.path.join(path, "niah_single_1_plot/example_0/kv_cache.pt")
    return path

def get_cache(path, args):
    cache_tensor = torch.load(path)
    cache = DynamicCache()
    if args.decode_strategy == "full":
        cache_tensor = cache_tensor.to("cuda")
    cache.key_cache = [t.unsqueeze(0) for t in list(cache_tensor[0])]
    cache.value_cache = [t.unsqueeze(0) for t in list(cache_tensor[1])]
    cache._seen_tokens = cache_tensor.shape[-2]
    if args.decode_strategy == "topk_flat":
        cache = TopkCache.from_dynamic_cache(cache)
    elif args.decode_strategy == "topk_ivf":
        raise NotImplementedError
    elif args.decode_strategy == "full":
        cache = cache.to("cuda")
    else:
        raise NotImplementedError

    return cache

def get_model(args):
    if args.decode_strategy in ["topk_flat", "topk_ivf"]:
        model = AutoTopkModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    return model

def get_query_tokens(dataset, tokenizer, device):
    # Hard-coded to select 1st example
    query_str = dataset[0]['input'] + dataset[0]['suffix']
    query_tokens = tokenizer(query_str, return_tensors="pt").to(device)
    return query_tokens

def time_generate(model, inputs, kwargs):
    model = model.eval()
    print(kwargs)
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            **kwargs,
        )
        elapsed_time = time.time() - start_time
    return elapsed_time 

def main():
    # Times generation of one token given a pre-built KV cache
    # Time per input token (decoding time)
    # Full?, H20, InfLLM (window attn), offloaded cache, vLLM, streamingLLM, flat index and IVF 
    args = get_args()

    # Currently only expect args.decode_strategy to be "full" or "topk_flat"
    if args.decode_strategy in ["offload", "topk_ivf", "streaming_llm"]:
        raise NotImplementedError
    dataset_path = os.path.join(args.dataset_dir, f"dataset_jsons/validation_{args.N}.jsonl")
    dataset = datasets.load_dataset("json", data_files={"eval": dataset_path})["eval"]
    cache_path = get_cache_path(args)
    cache = get_cache(cache_path, args)
    kwargs = get_kwargs(args, cache)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = get_model(args)
    query_inputs = get_query_tokens(dataset, tokenizer, model.device)
    total_time = time_generate(model, query_inputs, kwargs)
    tokens_per_second = kwargs['max_new_tokens'] / total_time 

    # Write output to file
    os.makedirs(args.output_dir, exist_ok=True)
    strategy_dir = os.path.join(args.output_dir, args.decode_strategy)
    os.makedirs(strategy_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(strategy_dir, f"timing_{args.decode_strategy}_N_{args.N}_{timestamp}.csv")
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        header = ["model", "decode_strategy", "N", "cache_length", "total_time", "tokens_per_second"]
        line = [args.model, args.decode_strategy, args.N, len(cache), total_time, tokens_per_second]
        if args.decode_strategy in ["topk_flat", "topk_ivf"]:
            header.append("k")
            line.append(args.k)
        writer.writerow(header)
        writer.writerow(line)

    # Implement len for topkcache
    
    print(f"Results saved to {output_file}")

if __name__=="__main__":
    main()
