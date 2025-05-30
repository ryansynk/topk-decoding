import os
import argparse
import time
import csv
import torch
import datasets
import sys
import traceback
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    DynamicCache,
    OffloadedCache,
    SinkCache,
    AutoTokenizer,
)
from topk_decoding import AutoTopkModelForCausalLM, TopkCache


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="gradientai/Llama-3-8B-Instruct-Gradient-1048k", type=str
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/fs/nexus-projects/FAST_Attention/ann",
        help="Path to dataset json files",
    )
    parser.add_argument(
        "--decode_strategy",
        type=str,
        choices=["full", "offloaded", "topk_flat", "topk_ivf", "topk_hnsw", "streaming_llm"],
        required=True,
    )
    parser.add_argument(
        "--nondeterministic",
        action="store_true",
        help="Turn on sampling during generation (output will be deterministic).",
    )
    parser.add_argument("--max_new_tokens", default=5, type=int)
    parser.add_argument(
        "--N",
        type=int,
        choices=[4096, 8192, 16384, 32768, 65536, 131072],
        required=True,
    )
    parser.add_argument("--k", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Name of GPU being used. Required for logging purposes",
    )
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
            temperature=None,
            top_p=None,
        )
    if args.decode_strategy in ["topk_flat", "topk_ivf", "topk_hnsw"]:
        kwargs["k"] = args.k
    kwargs["past_key_values"] = cache
    return kwargs


def get_cache_path(args):
    path = os.path.join(args.dataset_dir, args.model.split("/")[-1])
    path = os.path.join(path, str(args.N))
    path = os.path.join(path, "niah_single_1_plot/example_0/kv_cache.pt")
    return path


def get_pos_embs_path(args):
    path = os.path.join(args.dataset_dir, args.model.split("/")[-1])
    path = os.path.join(path, str(args.N))
    path = os.path.join(path, "niah_single_1_plot/example_0/pos_embs.pt")
    return path


def get_cache_full(cache_tensor, device):
    cache = DynamicCache()
    cache_tensor = cache_tensor.to(device)
    cache.key_cache = [t.unsqueeze(0) for t in list(cache_tensor[0])]
    cache.value_cache = [t.unsqueeze(0) for t in list(cache_tensor[1])]
    cache._seen_tokens = cache_tensor.shape[-2]
    cache = cache.to(device)
    return cache


def get_cache_offloaded(cache_tensor):
    cache = OffloadedCache()
    for l in range(cache_tensor.shape[1]):
        keys = cache_tensor[0, l, :, :, :].to("cuda").unsqueeze(0)
        values = cache_tensor[1, l, :, :, :].to("cuda").unsqueeze(0)
        cache.update(keys, values, l)
    return cache


def get_cache_streaming_llm(cache_tensor, pos_embs, k):
    if k < 5:
        num_sink_tokens = 1
    else:
        num_sink_tokens = 4
    window_length = k - num_sink_tokens
    cache = SinkCache(window_length=window_length, num_sink_tokens=num_sink_tokens)
    cos = pos_embs[0].to("cuda")
    sin = pos_embs[1].to("cuda")
    cache_kwargs = {"cos": cos, "sin": sin}
    for l in range(cache_tensor.shape[1]):
        keys = cache_tensor[0, l, :, :, :].to("cuda").unsqueeze(0)
        values = cache_tensor[1, l, :, :, :].to("cuda").unsqueeze(0)
        cache.update(keys, values, l, cache_kwargs)
    return cache


def get_cache(path, args):
    cache_tensor = torch.load(path)
    real_N = cache_tensor.shape[-2]

    if args.decode_strategy == "full":
        cache = get_cache_full(cache_tensor, "cuda")
    elif args.decode_strategy == "topk_flat":
        cache = get_cache_full(cache_tensor, "cpu")
        cache = TopkCache.from_dynamic_cache(cache)
    elif args.decode_strategy == "topk_ivf":
        ivf_path = os.path.join(os.path.split(path)[0], "ivf_cache")
        if os.path.isdir(ivf_path):
            cache = TopkCache.load(ivf_path)
        else:
            cache = get_cache_full(cache_tensor, "cpu")
            cache = TopkCache.from_dynamic_cache(cache, use_ivf=True)
            TopkCache.save(cache, ivf_path)
    elif args.decode_strategy == "topk_hnsw":
        hnsw_path = os.path.join(os.path.split(path)[0], "hnsw_cache")
        if os.path.isdir(hnsw_path):
            cache = TopkCache.load(hnsw_path)
        else:
            cache = get_cache_full(cache_tensor, "cpu")
            cache = TopkCache.from_dynamic_cache(cache, index_type="hnsw")
            TopkCache.save(cache, hnsw_path)
    elif args.decode_strategy == "offloaded":
        cache = get_cache_offloaded(cache_tensor)
    elif args.decode_strategy == "streaming_llm":
        pos_embs_path = get_pos_embs_path(args)
        pos_embs = torch.load(pos_embs_path)
        cache = get_cache_streaming_llm(cache_tensor, pos_embs, args.k)
    else:
        raise NotImplementedError

    return cache, real_N


def get_model(args):
    if args.decode_strategy in ["topk_flat", "topk_ivf", "topk_hnsw"]:
        model = AutoTopkModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
    return model


def get_query_tokens(dataset, tokenizer, device):
    # Hard-coded to select 1st example
    query_str = dataset[0]["input"] + dataset[0]["suffix"]
    query_tokens = tokenizer(query_str, return_tensors="pt").to(device)
    return query_tokens


def time_generate(model, inputs, kwargs):
    model = model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            **kwargs,
        )
        elapsed_time = time.time() - start_time
    return elapsed_time


def write_output(args, real_N, total_time, tokens_per_second, oom):
    os.makedirs(args.output_dir, exist_ok=True)
    strategy_dir = os.path.join(args.output_dir, args.decode_strategy)
    os.makedirs(strategy_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        strategy_dir, f"timing_{args.decode_strategy}_N_{args.N}_{timestamp}.csv"
    )
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        header = [
            "model",
            "decode_strategy",
            "N",
            "context_length",
            "total_time",
            "tokens_per_second",
            "oom",
            "device",
            "k",
        ]
        line = [
            args.model,
            args.decode_strategy,
            args.N,
            real_N,
            total_time,
            tokens_per_second,
            oom,
            args.device,
            args.k,
        ]
        writer.writerow(header)
        writer.writerow(line)
    print(f"Results saved to {output_file}")


def main():
    # Times generation of one token given a pre-built KV cache
    # Time per input token (decoding time)
    # Full?, H20, InfLLM (window attn), offloaded cache, vLLM, streamingLLM, flat index and IVF
    args = get_args()

    # Currently only expect args.decode_strategy to be "full" or "topk_flat"
    # if args.decode_strategy in ["streaming_llm"]:
    #    raise NotImplementedError
    dataset_path = os.path.join(
        args.dataset_dir, f"dataset_jsons/validation_{args.N}.jsonl"
    )
    dataset = datasets.load_dataset("json", data_files={"eval": dataset_path})["eval"]
    cache_path = get_cache_path(args)
    model = get_model(args)
    oom = False
    try:
        cache, real_N = get_cache(cache_path, args)
    except torch.cuda.OutOfMemoryError as e:
        print(f"Failed to move cache to GPU (OOM): {e}")
        cache = None
        real_N = -1
        total_time = None
        tokens_per_second = None
        oom = True

    if cache is not None:
        try:
            kwargs = get_kwargs(args, cache)
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            query_inputs = get_query_tokens(dataset, tokenizer, model.device)
            total_time = time_generate(model, query_inputs, kwargs)
            tokens_per_second = kwargs["max_new_tokens"] / total_time
        except torch.cuda.OutOfMemoryError as e:
            print(f"Generation failed, out of memory: {e}")
            total_time = None
            tokens_per_second = None
            oom = True
        except RuntimeError as e:
            print(f"Generation failed: {e}")
            total_time = None
            tokens_per_second = None
            oom = True
            sys.exit(traceback.format_exc())  # Print and exit

    # Write output to file
    write_output(args, real_N, total_time, tokens_per_second, oom)


if __name__ == "__main__":
    main()
