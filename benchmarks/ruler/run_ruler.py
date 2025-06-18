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
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Name of GPU being used. Required for logging purposes",
    )
    parser.add_argument(
        "--num_examples", type=int, default=None
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


def prefill_streaming_llm_cache(model, tokenizer, cache, context):
    print("ABout to prefill")
    with torch.no_grad():
        context_inputs = tokenizer(context, return_tensors="pt").to(model.device)
        length = context_inputs.input_ids.shape[-1]
        chunk_size = 124
        #num_chunks = 4
        #assert length % num_chunks == 0
        #chunk_size = int(length / num_chunks)
        tokens_processed = 0 
        i = 0
        while tokens_processed < length:
            ids = context_inputs.input_ids[:, i*chunk_size:(i+1)*chunk_size]
            cache = model(ids, past_key_values=cache).past_key_values
            tokens_processed = cache.get_seq_length()
            print(tokens_processed)
            print(cache.key_cache[0].shape)
            i = i + 1
    print("Finished prefill")
    return cache


def get_cache(path, args, model):
    cache_tensor = torch.load(path)
    real_N = cache_tensor.shape[-2]

    if args.decode_strategy == "full":
        cache = get_cache_full(cache_tensor, "cuda")
    elif args.decode_strategy == "topk_flat":
        assert args.k > 0, "Chose top-k but k not set!"
        cache = get_cache_full(cache_tensor, "cpu")
        cache = TopkCache.from_dynamic_cache(cache)
    elif args.decode_strategy == "topk_ivf":
        assert args.k > 0, "Chose top-k but k not set!"
        cache = get_cache_full(cache_tensor, "cpu")
        cache = TopkCache.from_dynamic_cache(cache, index_type="ivf")
    elif args.decode_strategy == "topk_hnsw":
        assert args.k > 0, "Chose top-k but k not set!"
        cache = get_cache_full(cache_tensor, "cpu")
        cache = TopkCache.from_dynamic_cache(cache, index_type="hnsw")
    elif args.decode_strategy == "offloaded":
        cache = get_cache_offloaded(cache_tensor)
    elif args.decode_strategy == "streaming_llm":
        # pos_embs_path = get_pos_embs_path(args)
        # pos_embs = torch.load(pos_embs_path)
        # cache = get_cache_streaming_llm(cache_tensor, pos_embs, args.k)
        if args.k < 5:
            num_sink_tokens = 1
        else:
            num_sink_tokens = 4
        window_length = args.k - num_sink_tokens
        cache = SinkCache(window_length=window_length, num_sink_tokens=num_sink_tokens)
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
        args.output_dir, f"{args.task}/{model_name_suffix}/N_{args.N}/{args.task}"
    )
    os.makedirs(output_path, exist_ok=True)
    if args.num_examples is not None:
        fname = f"answers_{args.decode_strategy}_nex_{args.num_examples}.csv"
    else:
        fname = f"answers_{args.decode_strategy}.csv"
    output_path = os.path.join(output_path, fname)
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

    query_delimiter_idx = decoded_output[0].rfind(query_delimiter) + len(
        query_delimiter
    )
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

    if args.num_examples is not None:
        dataset = dataset.select(range(args.num_examples))

    model = get_model(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    oom = False
    output_dataset = []
    for ex in tqdm(dataset):
        try:
            cache, real_N = get_cache(ex["context_cache_path"], args, model.model)
        except torch.cuda.OutOfMemoryError as e:
            cache = None
            real_N = -1
            total_time = None
            tokens_per_second = None
            oom = True
            predicted_output = [""] 
            correct = False

        if cache is not None:
            try:
                inputs = tokenizer(ex["context"] + ex["query"], return_tensors="pt").to(
                    model.device
                )
                if args.decode_strategy == "streaming_llm":
                    cache = prefill_streaming_llm_cache(
                        model, tokenizer, cache, ex["context"]
                    )
                kwargs = get_kwargs(args, cache, tokenizer)
                total_time, outputs = time_generate(model, inputs, kwargs)
                predicted_output = get_output_str(outputs, tokenizer, args.task)
                correct = predicted_output == ex["outputs"]
                num_tokens = outputs.shape[-1] - inputs.input_ids.shape[-1]
                tokens_per_second = num_tokens / total_time
            except torch.cuda.OutOfMemoryError as e:
                print(f"Generation failed, out of memory: {e}")
                total_time = None
                tokens_per_second = None
                correct = False
                predicted_output = [""] 
                oom = True
                num_tokens = None
            except RuntimeError as e:
                print(f"Generation failed: {e}")
                total_time = None
                tokens_per_second = None
                oom = True
                correct = False
                predicted_output = [""] 
                num_tokens = None
                sys.exit(traceback.format_exc())  # Print and exit

        ex.update(
            {
                "total_time": total_time,
                "tokens_per_second": tokens_per_second,
                "oom": oom,
                "predicted_output": predicted_output[0],
                "correct": correct,
                "outputs": ex["outputs"][0],
                "peak_gpu_memory": torch.cuda.max_memory_allocated(),
                "k": args.k,
                "decode_strategy": args.decode_strategy,
                "num_tokens": num_tokens,
            }
        )
        output_dataset.append(ex)
        torch.cuda.reset_max_memory_allocated()

    # Write output to file
    write_output(args, output_dataset)


if __name__ == "__main__":
    main()
