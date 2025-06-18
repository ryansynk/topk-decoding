import os
import argparse
import torch
import datasets
import polars as pl
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    DynamicCache,
    AutoTokenizer,
)
from topk_decoding import AutoTopkModelForCausalLM, TopkCache


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="gradientai/Llama-3-8B-Instruct-1048k", type=str
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data",
        help="Path to dataset json files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to place output csvs into",
    )
    parser.add_argument(
        "--N",
        type=int,
        choices=[32768, 65536, 131072],
        required=True,
    )
    parser.add_argument("--task", type=str, default="niah_multikey_1")
    parser.add_argument(
        "--decode_strategy",
        type=str,
        choices=[
            "topk_flat",
            "topk_ivf",
            "topk_hnsw",
        ],
        required=True,
    )
    parser.add_argument(
        "--nondeterministic",
        action="store_true",
        help="Turn on sampling during generation (output will be deterministic).",
    )
    parser.add_argument("--max_new_tokens", default=5, type=int)
    parser.add_argument("--k", type=int, default=-1)
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Name of GPU being used. Required for logging purposes",
    )
    parser.add_argument("--num_examples", type=int, default=15)
    return parser.parse_args()


def get_kwargs(args, cache, tokenizer):
    if args.nondeterministic:
        kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            top_p=1,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    else:
        kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=1,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    if args.decode_strategy in ["topk_flat", "topk_ivf", "topk_hnsw"]:
        kwargs["k"] = args.k
    kwargs["past_key_values"] = cache
    return kwargs


def get_cache_full(cache_tensor, device):
    cache = DynamicCache()
    cache_tensor = cache_tensor.to(device)
    cache.key_cache = [t.unsqueeze(0) for t in list(cache_tensor[0])]
    cache.value_cache = [t.unsqueeze(0) for t in list(cache_tensor[1])]
    cache._seen_tokens = cache_tensor.shape[-2]
    cache = cache.to(device)
    return cache


def get_cache(path, args, model):
    cache_tensor = torch.load(path)
    real_N = cache_tensor.shape[-2]

    if args.decode_strategy == "topk_flat":
        assert args.k > 0, "Chose top-k but k not set!"
        cache = get_cache_full(cache_tensor, "cpu")
        cache = TopkCache.from_dynamic_cache(cache)
    elif args.decode_strategy == "topk_ivf":
        assert args.k > 0, "Chose top-k but k not set!"
        cache = get_cache_full(cache_tensor, "cpu")
        cache = TopkCache.from_dynamic_cache(cache, use_ivf=True)
    elif args.decode_strategy == "topk_hnsw":
        assert args.k > 0, "Chose top-k but k not set!"
        cache = get_cache_full(cache_tensor, "cpu")
        cache = TopkCache.from_dynamic_cache(cache, index_type="hnsw")
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
        raise NotImplementedError
    return model


def profile_generate(model, inputs, kwargs):
    model = model.eval()
    with torch.no_grad():
        with torch.profiler.record_function("generate"):
            outputs = model.generate(
                **inputs,
                **kwargs,
            )


def get_dataset_path(args):
    path = os.path.join(args.dataset_dir, f"N_{args.N}")
    path = os.path.join(path, args.task)
    model_suffix = args.model.split("/")[-1]
    path = os.path.join(path, model_suffix)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "validation_w_cache.jsonl")
    return os.path.abspath(path)

def get_time_fractions(prof_dict):
    time_fractions_dict = {}
    other = prof_dict["generate"] - (prof_dict["search"] + prof_dict["QToCPU"] + prof_dict["OutputToGPU"])
    time_fractions_dict["total"] = prof_dict["generate"] / prof_dict["generate"]
    time_fractions_dict["movement"] = (prof_dict["QToCPU"] + prof_dict["OutputToGPU"]) / prof_dict["generate"]
    time_fractions_dict["search"] = prof_dict["search"] / prof_dict["generate"]
    time_fractions_dict["other"] = other / prof_dict["generate"]
    return time_fractions_dict


def main():
    args = get_args()
    dataset_path = get_dataset_path(args)
    dataset = datasets.load_dataset("json", data_files=dataset_path)["train"]
    dataset = dataset.select(range(args.num_examples))

    model = get_model(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    output_dataset = []
    tracked_labels = [
        "generate",
        "QToCPU",
        "search",
        "OutputToGPU"
    ]

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/"),
        with_stack=True,  # Capture stack information for more context
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for ex in tqdm(dataset):
            cache, real_N = get_cache(ex["context_cache_path"], args, model.model)
            inputs = tokenizer(ex["context"] + ex["query"], return_tensors="pt").to(
                model.device
            )
            kwargs = get_kwargs(args, cache, tokenizer)
            profile_generate(model, inputs, kwargs)
            prof.step()
            torch.cuda.reset_max_memory_allocated()

    cpu_times_total_by_label = {label: 0.0 for label in tracked_labels}
    gpu_times_total_by_label = {label: 0.0 for label in tracked_labels}

    # Iterate through the profiler's aggregated results
    # prof.key_averages() returns a list of FunctionEvent objects
    for event in prof.key_averages():
        if event.key in tracked_labels:
            # We are interested in the 'self' time, which is time spent
            # exclusively within this operation, excluding its children.
            # 'cuda_time_total' is in microseconds, convert to milliseconds.
            # 'cpu_time_total' is in microseconds, convert to milliseconds.
            cpu_times_total_by_label[event.key] += event.cpu_time_total / 1000.0
            gpu_times_total_by_label[event.key] += event.device_time_total / 1000.0

    cpu_time_fractions = get_time_fractions(cpu_times_total_by_label)
    gpu_time_fractions = get_time_fractions(gpu_times_total_by_label)
    model_name_suffix = args.model.split("/")[-1]
    output_path = os.path.join(
        args.output_dir, f"{args.task}/{model_name_suffix}/N_{args.N}/prof.txt"
    )

    with open(output_path, 'w') as f:
        print(f"Total num examples = {args.num_examples}", file=f)
        print(f"\nDecode strategy = {args.decode_strategy}", file=f)
        print("\ncpu_time_total spent in tracked functions (across all steps):", file=f)
        for label, cum_time_ms in cpu_times_total_by_label.items():
            print(f"  - {label}: {cum_time_ms:.3f} ms", file=f)
        print("\ndevice_time_total spent in tracked functions (across all steps):", file=f)
        for label, cum_time_ms in gpu_times_total_by_label.items():
            print(f"  - {label}: {cum_time_ms:.3f} ms", file=f)
        print("\nFraction of Decode CPU Time Spent:", file=f)
        for label, percent in cpu_time_fractions.items():
            print(f"  - {label}: {percent:.3f} %", file=f)
        print("\nfractions of Decode GPU Time Spent:", file=f)
        for label, percent in gpu_time_fractions.items():
            print(f"  - {label}: {percent:.3f} %", file=f)


if __name__ == "__main__":
    main()
