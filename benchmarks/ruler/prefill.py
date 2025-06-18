import argparse
import os
import functools
import gc
import datasets
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, OffloadedCache
from topk_decoding import AutoTopkModelForCausalLM


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
        "--cache_dir",
        type=str,
        default="/fs/nexus-projects/topk_attention/kv_caches/colm_rebuttal/timing/",
    )
    parser.add_argument(
        "--N",
        type=int,
        choices=[32768, 65536, 131072],
        required=True,
    )
    parser.add_argument("--task", type=str, default="niah_multikey_1")
    parser.add_argument(
        "--unroll", action="store_true", help="Turns on MLP unrolling to save memory"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=-1,
        help="Select a subset of total examples (100 max)",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Index of dataset to start on"
    )
    return parser.parse_args()


def get_dataset_path(args):
    path = os.path.join(args.dataset_dir, f"N_{args.N}")
    path = os.path.join(path, args.task)
    path = os.path.join(path, "validation.jsonl")
    return os.path.abspath(path)


def get_output_path(args):
    path = os.path.join(args.dataset_dir, f"N_{args.N}")
    path = os.path.join(path, args.task)
    model_suffix = args.model.split("/")[-1]
    path = os.path.join(path, model_suffix)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "validation_w_cache.jsonl")
    return os.path.abspath(path)


def get_context_query(ex, task):
    if task in ["qa_1", "qa_2"]:
        query_delimiter = "\n\nQuestion:"
    else:
        query_delimiter = "[/INST]"
    text = ex["input"]
    query_delimiter_idx = text.rfind(query_delimiter) + len(query_delimiter)
    context = text[:query_delimiter_idx]
    query = text[query_delimiter_idx:]
    ex.update({"context": context, "query": query})
    return ex


def main():
    args = get_args()
    dataset_path = get_dataset_path(args)
    output_path = get_output_path(args)
    model_suffix = args.model.split("/")[-1]
    cache_path = os.path.join(args.cache_dir, f"{args.task}/{model_suffix}/{args.N}")
    os.makedirs(cache_path, exist_ok=True)
    dataset = datasets.load_dataset("json", data_files={"eval": dataset_path})["eval"]
    if args.num_examples > 0:
        dataset = dataset.select(range(args.start_idx, args.start_idx + args.num_examples))
    else:
        dataset = dataset.select(range(args.start_idx, 100))
    dataset = dataset.map(functools.partial(get_context_query, task=args.task))

    if args.unroll:
        model = AutoTopkModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            mlp=True,
            attn=False,
        ).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")
    model = model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dataset_w_caches = []
    for ex in tqdm(dataset):
        with torch.no_grad():
            # Prefills cache with context
            cache = OffloadedCache()
            context_inputs = tokenizer(ex["context"], return_tensors="pt").to(
                model.device
            )
            cache = model(**context_inputs, past_key_values=cache).past_key_values

            # Converts cache to tensor
            (H, N, D) = cache.key_cache[0].squeeze().shape
            dtype = cache.key_cache[0].dtype
            cache_tensor = torch.zeros((2, 32, H, N, D), dtype=dtype)
            for l, (k, v) in enumerate(zip(cache.key_cache, cache.value_cache)):
                cache_tensor[0, l, ...] = k.cpu().squeeze()
                cache_tensor[1, l, ...] = v.cpu().squeeze()

            # Saves tensor to file
            index = ex["index"]
            example_path = os.path.join(cache_path, f"example_{index}")
            os.makedirs(example_path, exist_ok=True)
            example_path = os.path.join(example_path, f"kv_cache.pt")
            torch.save(cache_tensor, example_path)
            ex.update({"context_cache_path": example_path})
            dataset_w_caches.append(ex)
            torch.cuda.empty_cache()
            gc.collect()

    dataset_w_caches = datasets.Dataset.from_list(dataset_w_caches)
    dataset_w_caches.to_json(output_path, orient="records")


if __name__ == "__main__":
    main()
