# project_setup

A small harness for trying different attention methods and models against the
[topk_decoding](../src/topk_decoding/) library without editing source code.

## Quick start

From the repo root:

```bash
uv run python -m project_setup.run --model tiny-llama --attn knn-flat --k 16 --input "Hello world"
```

That runs the tiny synthetic LLaMA with exact top-k FAISS attention on the input
"Hello world", asking for the top 16 keys each step.

## Available models

| Name | What it is | Cost |
|---|---|---|
| `tiny-llama` | Random-init 2-layer LLaMA built from scratch (~115k params, vocab 256, byte tokenizer). | None — runs on CPU, no download. |
| `llama-3-8b-1048k` | The production `gradientai/Llama-3-8B-Instruct-1048k`. | ~16 GB download, requires GPU. |

## Available attention methods

| Name | What it is | Exactness |
|---|---|---|
| `dense` | Standard attention. Every query attends to every past key. | Exact. |
| `knn-flat` | Exact top-k via FAISS `IndexFlatIP` (brute-force inner product). | Exact within top-k. |
| `ann-ivf` | Approximate top-k via FAISS `IndexIVFFlat` (inverted file index). | Approximate. |
| `ann-hnsw` | Approximate top-k via FAISS `IndexHNSWFlat` (small-world graph). | Approximate. |

## CLI flags

```
--model           tiny-llama | llama-3-8b-1048k   (required)
--attn            dense | knn-flat | ann-ivf | ann-hnsw   (required)
--k               int   top-k value; ignored for --attn dense   (default: 32)
--input           str   input text   (default: "Hello world")
--max-new-tokens  int   how many tokens to generate   (default: 5)
```

## Running the test

One correctness check ships here: `knn-flat` with `k = context_length` must produce
the same output as `dense`. FAISS Flat is exact, so asking for the top-k of all keys
returns every key — the math reduces to dense attention.

```bash
uv run pytest project_setup/test_attention.py -v
```

Runs in ~8 seconds on CPU.

## Adding a new attention method

1. Add an entry to `ATTENTION_METHODS` in [attention_methods.py](attention_methods.py)
   with `use_topk` and (if applicable) an `index_type` recognized by
   [`TopkCache.create_key_database`](../src/topk_decoding/topk_cache.py).
2. That's it. The new name immediately works in `run.py` via `--attn <new-name>`.

## Adding a new model

1. Write a loader function in [models.py](models.py) with signature
   `(use_topk: bool) -> (model, tokenizer)`. The model should be a standard
   `LlamaForCausalLM` (other architectures aren't supported by the library today).
2. Register it in the `MODELS` dict.

## Notes

- `tiny-llama` uses a byte tokenizer (each byte → one token, vocab 256). Generated
  output is decoded as UTF-8 with replacement, so non-printable bytes show as
  garbage. That's fine — the point is the swap mechanism, not the prose quality.
- `ann-ivf` needs enough keys to train its inverted file. Short inputs may hit the
  degenerate single-cluster case.
