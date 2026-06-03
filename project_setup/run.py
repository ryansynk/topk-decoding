"""CLI entry point for the swap-and-run harness.

Example:
    uv run python -m project_setup.run --model tiny-llama --attn knn-flat --k 16 \\
        --input "Hello world" --max-new-tokens 5
"""

import argparse

import torch

from project_setup.attention_methods import ATTENTION_METHODS, generate_kwargs
from project_setup.models import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--attn", required=True, choices=list(ATTENTION_METHODS.keys()))
    parser.add_argument("--k", type=int, default=32, help="Top-k value (ignored for --attn dense).")
    parser.add_argument("--input", default="Hello world", help="Input text to feed the model.")
    parser.add_argument("--max-new-tokens", type=int, default=5)
    args = parser.parse_args()

    method = ATTENTION_METHODS[args.attn]
    print(f"[load] model={args.model} use_topk={method.use_topk}")
    model, tokenizer = MODELS[args.model](use_topk=method.use_topk)

    enc = tokenizer(args.input, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    print(f"[tokenize] {input_ids.shape[-1]} tokens")

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        **generate_kwargs(method, args.k),
    }
    k_label = args.k if method.use_topk else "n/a"
    print(f"[generate] attn={args.attn} k={k_label}")

    with torch.no_grad():
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

    seq = output.sequences[0] if hasattr(output, "sequences") else output[0]
    decoded = tokenizer.decode(seq)
    print("---")
    print(decoded)


if __name__ == "__main__":
    main()
