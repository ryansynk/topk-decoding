import os
import argparse
import functools
import time
import csv
import torch
import datasets
import sys
import traceback
import re
import polars as pl
from datetime import datetime
from tqdm import tqdm
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
            "full",
            "offloaded",
            "topk_flat",
            "topk_ivf",
            "topk_hnsw",
            "streaming_llm",
        ],
        required=True,
    )
    parser.add_argument(
        "--nondeterministic",
        action="store_true",
        help="Turn on sampling during generation (output will be deterministic).",
    )
    parser.add_argument("--max_new_tokens", default=5, type=int)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Name of GPU being used. Required for logging purposes",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Runs in debug mode with 2 examples"
    )
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


def get_suffix_index(tokenizer, text, query_delimiter="[/INST]"):
    # Return the index of the first token after the string "[/INST]" in the prompt. This is used to split the prompt into prefix and suffix for topk attention.
    # get the sequence of ids that makes up an instruction ending token. Tokenizers always start with a beginning token so strip that off.
    inst_token_ids = tokenizer(
        [query_delimiter], return_tensors="pt", padding=False
    ).input_ids[0, 1:]
    prompt_token_ids = tokenizer(text, return_tensors="pt", padding=True).input_ids[0]
    prompt_length = len(prompt_token_ids)
    inst_token_length = len(inst_token_ids)
    for i in range(prompt_length - inst_token_length + 1):
        if torch.equal(prompt_token_ids[i : i + inst_token_length], inst_token_ids):
            return i + inst_token_length
    return -1


def get_suffix_indices(example, tokenizer):
    query_delimiter = "[/INST]"
    text = example["input"]
    suffix_text_idx = text.rfind(query_delimiter) + len(query_delimiter)
    suffix_token_idx = get_suffix_index(tokenizer, text, query_delimiter)
    assert (
        suffix_text_idx != -1 and suffix_token_idx != 1
    ), "Instruction token not found in prompt"
    # kv_cache_prefix = kv_cache[:, :, :, :suffix_token_idx, :]
    example.update(
        {
            "suffix_token_idx": suffix_token_idx,
            "suffix_text": text[suffix_text_idx:],
        }
    )
    return example


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


def get_cache(path, args, model):
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
           config = model.config
           cache = TopkCache.load(ivf_path, config.num_hidden_layers, config.num_key_value_heads)
        else:
           cache = get_cache_full(cache_tensor, "cpu")
           cache = TopkCache.from_dynamic_cache(cache, use_ivf=True)
           TopkCache.save(cache, ivf_path)
        cache = get_cache_full(cache_tensor, "cpu")
        cache = TopkCache.from_dynamic_cache(cache, use_ivf=True)
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


# def get_query_tokens(dataset, tokenizer, device):
#    # Hard-coded to select 1st example
#    query_str = dataset[0]["input"] + dataset[0]["suffix"]
#    query_tokens = tokenizer(query_str, return_tensors="pt").to(device)
#    return query_tokens


def time_generate(model, inputs, kwargs):
    model = model.eval()
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            **kwargs,
        )
        elapsed_time = time.time() - start_time
    return elapsed_time, outputs

def write_output(args, output_dataset):
    model_name_suffix = args.model.split("/")[-1]
    output_path = os.path.join(
        args.output_dir,
        f"{args.task}/{model_name_suffix}/N_{args.N}/{args.task}"
    )
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(
        output_path,
        f"answers_{args.decode_strategy}.csv"
    )
    df = pl.from_dicts(output_dataset)
    df.write_csv(output_path)

    print(f"Results saved to {output_path}")


def get_output_str(outputs, tokenizer, task):
    decoded_output = tokenizer.batch_decode(
        outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    if task in ["qa_1", "qa_2"]:
        query_delimiter = "\n\nQuestion:"
    else:
        query_delimiter = "[/INST]"

    query_delimiter_idx = decoded_output[0].rfind(query_delimiter) + len(query_delimiter)
    answer_str = decoded_output[0][query_delimiter_idx:]
    match = re.search(r"\b\d+\b", answer_str)
    if match:
        output_str = [match.group()]
    else:
        output_str = [""]
    return output_str

def get_dataset_path(args):
    path = os.path.join(args.dataset_dir, f"N_{args.N}")
    path = os.path.join(path, args.task)
    model_suffix = args.model.split("/")[-1]
    path = os.path.join(path, model_suffix)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "validation_w_cache.jsonl")
    return os.path.abspath(path)

def main():
    # Times generation of one token given a pre-built KV cache
    # Time per input token (decoding time)
    # Full?, H20, InfLLM (window attn), offloaded cache, vLLM, streamingLLM, flat index and IVF
    args = get_args()
    dataset_path = get_dataset_path(args)
    dataset = datasets.load_dataset("json", data_files=dataset_path)["train"]

    if args.debug:
        dataset = dataset.select(range(2))

    model = get_model(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    oom = False
    output_dataset = []
    for ex in tqdm(dataset):
        try:
            cache, real_N = get_cache(
                ex["context_cache_path"], args, model.model
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"Failed to move cache to GPU (OOM): {e}")
            cache = None
            real_N = -1
            total_time = None
            tokens_per_second = None
            oom = True
            predicted_output = None
            correct = False

        if cache is not None:
            try:
                inputs = tokenizer(ex['context'] + ex['query'], return_tensors="pt").to(model.device)
                kwargs = get_kwargs(args, cache, tokenizer)
                total_time, outputs = time_generate(model, inputs, kwargs)
                predicted_output = get_output_str(outputs, tokenizer, args.task)
                correct = predicted_output == ex["outputs"]
                tokens_per_second = kwargs["max_new_tokens"] / total_time
            except torch.cuda.OutOfMemoryError as e:
                print(f"Generation failed, out of memory: {e}")
                total_time = None
                tokens_per_second = None
                correct = False
                predicted_output = None
                oom = True
            except RuntimeError as e:
                print(f"Generation failed: {e}")
                total_time = None
                tokens_per_second = None
                oom = True
                correct = False
                predicted_output = None
                sys.exit(traceback.format_exc())  # Print and exit

        ex.update(
            {
                "total_time": total_time,
                "tokens_per_second": tokens_per_second,
                "oom": oom,
                "predicted_output": predicted_output[0],
                "correct": correct,
                "outputs": ex["outputs"][0],
                "peak_gpu_memory": torch.cuda.max_memory_allocated()
            }
        )
        output_dataset.append(ex)
        torch.cuda.reset_max_memory_allocated()

    # Write output to file
    write_output(args, output_dataset)


if __name__ == "__main__":
    main()
