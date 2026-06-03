"""Registry of attention methods you can swap between.

Each entry describes (a) whether the model needs top-k patching, and (b) which
FAISS backend to use during generation. The actual wiring happens in `run.py`.

To add a new method: append an entry to `ATTENTION_METHODS`.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AttentionMethod:
    use_topk: bool
    index_type: Optional[str] = None
    description: str = ""


ATTENTION_METHODS = {
    "dense": AttentionMethod(
        use_topk=False,
        description="Standard dense attention. Every query attends to every past key.",
    ),
    "knn-flat": AttentionMethod(
        use_topk=True,
        index_type="flat",
        description="Exact top-k via FAISS IndexFlatIP (brute-force inner product).",
    ),
    "ann-ivf": AttentionMethod(
        use_topk=True,
        index_type="ivf",
        description="Approximate top-k via FAISS IndexIVFFlat (inverted file index).",
    ),
    "ann-hnsw": AttentionMethod(
        use_topk=True,
        index_type="hnsw",
        description="Approximate top-k via FAISS IndexHNSWFlat (small-world graph).",
    ),
}


def generate_kwargs(method: AttentionMethod, k: int) -> dict:
    """Build the kwargs to pass to model.generate() for this method."""
    if not method.use_topk:
        return {}
    return {"k": k, "index_type": method.index_type}
