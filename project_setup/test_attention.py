"""Correctness anchor: with k = context_length, knn-flat must match dense exactly.

Rationale: FAISS IndexFlatIP is exact brute-force inner-product search. When we ask
it for the top-k of N keys with k >= N, it returns all N keys. The top-k attention
math then reduces to dense attention, so the output should match bit-for-bit.

This test runs on a tiny synthetic LLaMA so it stays CPU-friendly and fast.
"""

import torch

from project_setup.attention_methods import ATTENTION_METHODS, generate_kwargs
from project_setup.models import load_tiny_llama


def _generate_with_method(method_name: str, input_ids, attention_mask, k, seed=42):
    method = ATTENTION_METHODS[method_name]
    # Re-seed before model construction so weights are identical across calls.
    torch.manual_seed(seed)
    model, _ = load_tiny_llama(use_topk=method.use_topk)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=3,
        do_sample=False,
        **generate_kwargs(method, k=k),
    )
    return out.sequences[0] if hasattr(out, "sequences") else out[0]


def test_knn_flat_matches_dense_at_full_k():
    torch.manual_seed(0)
    input_ids = torch.randint(0, 256, (1, 16), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    k = input_ids.shape[-1]  # ask for top-k of all keys → exact

    dense_seq = _generate_with_method("dense", input_ids, attention_mask, k=k)
    knn_seq = _generate_with_method("knn-flat", input_ids, attention_mask, k=k)

    assert torch.equal(dense_seq, knn_seq), (
        f"\nDENSE generated: {dense_seq.tolist()}"
        f"\nKNN   generated: {knn_seq.tolist()}"
    )
