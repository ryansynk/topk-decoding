from dataclasses import dataclass
from typing import Any, Dict, DefaultDict, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path

import math
import os
import torch
import faiss
import einops
from transformers import Cache, DynamicCache


class TopkCache(Cache):
    """
    Structure of the cache:
    - Two fields, key_cache and value_cache.
        - Each field is a list, with one item per layer of the model
            - Each item is a tuple, with two elements.
                - The first element contains keys/values from the prefix (construct phase)
                - The second element contains keys/values from the suffix (query/generation phase)

    Example:
        self.key_cache:
            [
                (faiss.swigfaiss_avx2.IndexFlatIP, torch.Tensor),
                ...
                (faiss.swigfaiss_avx2.IndexFlatIP, torch.Tensor)
            ]
        self.value_cache:
            [
                (torch.Tensor, torch.Tensor),
                ...
                (torch.Tensor, torch.Tensor)
            ]
    """

    def __init__(self, flat: bool = True) -> None:
        self.key_cache: List[(faiss.swigfaiss_avx2.IndexFlatIP, torch.Tensor)] = []
        self.value_cache: List[(torch.Tensor, torch.Tensor)] = []
        self.dense_key_cache: List[torch.Tensor] = []
        self.dense_value_cache: List[torch.Tensor] = []
        self.seq_lengths: DefaultDict[int, int] = defaultdict(int)
        self.sparse_cache_initialized: DefaultDict[int, bool] = defaultdict(bool)
        self.flat = flat
        super().__init__()

    def __len__(self) -> int:
        return self.get_seq_length()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. These are specific to each subclass and allow new types of
                cache to be created.

        Return:
            A tuple containing the updated key and value states.
        """
        self.seq_lengths[layer_idx] += key_states.shape[-2]
        # We update the suffix part of the cache (the second part of each tuple)
        assert len(self.key_cache) > layer_idx
        self.key_cache[layer_idx] = (
            self.key_cache[layer_idx][0],
            torch.cat([self.key_cache[layer_idx][1], key_states], dim=-2),
        )
        self.value_cache[layer_idx] = (
            self.value_cache[layer_idx][0],
            torch.cat([self.value_cache[layer_idx][1], value_states], dim=-2),
        )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return self.seq_lengths[layer_idx]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states, if there is any."""
        return None

    def to_legacy_cache(self):
        return self

    def update_dense_to_sparse(self):
        for layer_idx, key_cache_entry in enumerate(self.key_cache):
            sparse_prefix_keys, _ = key_cache_entry
            dense_keys = self.dense_key_cache[layer_idx]
            prefix_key_cache_update = torch.cat(
                (sparse_prefix_keys, dense_keys.cpu()), dim=-2
            )
            self.key_cache[layer_idx] = (
                prefix_key_cache_update,
                torch.empty(0).cuda().to(dense_keys.dtype),
            )

        for layer_idx, value_cache_entry in enumerate(self.value_cache):
            sparse_prefix_values, _ = value_cache_entry
            dense_values = self.dense_value_cache[layer_idx]
            prefix_value_cache_update = torch.cat(
                (sparse_prefix_values, dense_values.cpu()), dim=-2
            )
            self.value_cache[layer_idx] = (
                prefix_value_cache_update,
                torch.empty(0).cuda().to(dense_values.dtype),
            )

        self.dense_key_cache = []
        self.dense_value_cache = []

    def update_tensor_to_faiss_index(self):
        for layer_idx, key_cache_entry in enumerate(self.key_cache):
            sparse_prefix_keys, suffix_keys = key_cache_entry
            key_db = DynamicFaissCache.create_key_database(
                flat=True, key_states=sparse_prefix_keys
            )
            self.key_cache[layer_idx] = (key_db, suffix_keys)

    @staticmethod
    def create_key_database(key_states, flat=True, BH=None, D=None):
        """Create a key vector database using FAISS. Stored on CPU.

        Args:
            key_states (torch.Tensor): Tensor of key states.

        Returns:
            list: List of FAISS search indexes, one for each batch and head.
        """
        assert key_states is not None

        if len(key_states.shape) == 4:
            B, H, N, D = key_states.shape
            BH = B * H
            key_states = einops.rearrange(key_states, "B H N D -> (B H) N D")
        else:
            BH, N, D = key_states.shape

        # TODO parallelize?
        faiss_indices_list = []
        key_database = []
        for i in range(BH):
            if flat:
                search_index = faiss.IndexFlatIP(D)
                search_index.add(
                    key_states[i, :, :]
                    .contiguous()
                    .to(torch.float32)
                    .cpu()
                    .detach()
                    .numpy()
                )
            else:
                quantizer = faiss.IndexFlatIP(D)
                if N > 900000:
                    nlists = int(math.ceil(math.sqrt(N)))
                else:
                    nlists = int(math.ceil(N / 1000))

                # TODO have some way of setting this?
                nprobes = 8

                search_index = faiss.IndexIVFFlat(
                    quantizer, D, nlists, faiss.METRIC_INNER_PRODUCT
                )
                if N < 100000:
                    search_index.train(
                        key_states[i, :, :].contiguous().to(torch.float32).cpu()
                    )
                else:
                    search_index.train(
                        key_states[i, :100000, :].contiguous().to(torch.float32).cpu()
                    )
                search_index.add(
                    key_states[i, :, :].contiguous().to(torch.float32).cpu()
                )
                search_index.nprobe = nprobes

            key_database.append(search_index)

        return key_database

    @staticmethod
    def create_key_database_new(key_states, index_type, BH=None, D=None):
        """Create a key vector database using FAISS. Stored on CPU.

        Args:
            key_states (torch.Tensor): Tensor of key states.

        Returns:
            list: List of FAISS search indexes, one for each batch and head.
        """
        assert key_states is not None
        index_types = ["flat", "ivf", "hnsw"]

        if len(key_states.shape) == 4:
            B, H, N, D = key_states.shape
            BH = B * H
            key_states = einops.rearrange(key_states, "B H N D -> (B H) N D")
        else:
            BH, N, D = key_states.shape

        # TODO parallelize?
        faiss_indices_list = []
        key_database = []
        key_states = key_states.cpu().to(torch.float32)
        for i in range(BH):
            head_key_states = key_states[i, :, :].contiguous().detach().numpy()
            if index_type=="flat":
                search_index = faiss.IndexFlatIP(D)
                search_index.add(
                    head_key_states
                )
            elif index_type=="ivf":
                quantizer = faiss.IndexFlatIP(D)
                if N > 900000:
                    nlists = int(math.ceil(math.sqrt(N)))
                else:
                    nlists = int(math.ceil(N / 1000))

                # TODO have some way of setting this?
                nprobes = 8

                search_index = faiss.IndexIVFFlat(
                    quantizer, D, nlists, faiss.METRIC_INNER_PRODUCT
                )
                if N < 100000:
                    search_index.train(
                        head_key_states
                    )
                else:
                    search_index.train(
                        head_key_states[:100000, :]
                    )
                search_index.add(
                    head_key_states
                )
                search_index.nprobe = nprobes
            elif index_type=="hnsw":
                M = 16
                search_index = faiss.IndexHNSWFlat(D, M, faiss.METRIC_INNER_PRODUCT)
                search_index.add(
                    head_key_states
                )
            else:
                raise ValueError("Excpeted index_type to be one of: {}, instead got {index_type}")

            key_database.append(search_index)

        return key_database

    @staticmethod
    def update_key_database(key_states, key_db):
        """Adds key_states to a given faiss key vector database.

        Args:
            key_states (torch.Tensor): Tensor of key states.
            key_db (faiss index): Current key database

        Returns:
            list: List of FAISS search indexes, one for each batch and head.
        """
        if len(key_states.shape) == 4:
            B, H, N, D = key_states.shape
            BH = B * H
            key_states = einops.rearrange(key_states, "B H N D -> (B H) N D")
        else:
            BH, N, D = key_states.shape

        # TODO parallelize?
        faiss_indices_list = []
        key_database = []
        for i, db in enumerate(key_db):
            db.add(key_states[i, :, :].contiguous().to(torch.float32).cpu())

        return key_db

    @staticmethod
    def save(cache: "DynamicFaissCache", directory: str) -> None:
        cache_path = Path(directory)
        cache_path.mkdir(parents=True, exist_ok=True)
        key_cache_path = cache_path.joinpath("key_cache")
        key_cache_path.mkdir(parents=True, exist_ok=True)
        value_cache_path = cache_path.joinpath("value_cache")
        value_cache_path.mkdir(parents=True, exist_ok=True)

        layer_idx = 0
        for prefix_key_db, suffix_key_states in cache.key_cache:
            prefix_key_indices_path = key_cache_path.joinpath(
                f"layer_{layer_idx}_indices"
            )
            prefix_key_indices_path.mkdir(parents=True, exist_ok=True)
            bh_num = 0
            for db in prefix_key_db:
                prefix_key_index_path = prefix_key_indices_path.joinpath(
                    f"head_{bh_num}.index"
                )
                faiss.write_index(
                    prefix_key_db[bh_num], str(prefix_key_index_path.absolute())
                )
                bh_num = bh_num + 1
            layer_idx = layer_idx + 1

        layer_idx = 0
        for prefix_value_states, suffix_value_states in cache.value_cache:
            prefix_value_tensor_path = value_cache_path.joinpath(
                f"layer_{layer_idx}_prefix.pt"
            )
            torch.save(prefix_value_states, str(prefix_value_tensor_path.absolute()))
            layer_idx = layer_idx + 1

    @staticmethod
    def load(
        directory: str,
        num_layers: int,
        bh_size: int,
        device="cuda",
        dtype=torch.bfloat16,
    ) -> "DynamicFaissCache":
        cache_path = Path(directory)
        key_cache_path = cache_path.joinpath("key_cache")
        value_cache_path = cache_path.joinpath("value_cache")

        key_cache = []
        value_cache = []
        seq_lengths = defaultdict(int)

        for layer_idx in range(num_layers):
            prefix_key_indices_path = key_cache_path.joinpath(
                f"layer_{layer_idx}_indices"
            )
            key_database = []
            for bh_num in range(bh_size):
                prefix_key_index_path = prefix_key_indices_path.joinpath(
                    f"head_{bh_num}.index"
                )
                key_database.append(
                    faiss.read_index(str(prefix_key_index_path.absolute()))
                )
            key_cache.append((key_database, torch.empty(0).to(device).to(dtype)))
            seq_lengths[layer_idx] = key_database[0].ntotal

        for layer_idx in range(num_layers):
            prefix_value_tensor_path = value_cache_path.joinpath(
                f"layer_{layer_idx}_prefix.pt"
            )
            prefix_value_states = torch.load(str(prefix_value_tensor_path.absolute()))
            value_cache.append(
                (prefix_value_states, torch.empty(0).to(device).to(dtype))
            )

        cache = DynamicFaissCache()
        cache.key_cache = key_cache
        cache.value_cache = value_cache
        cache.seq_lengths = seq_lengths
        return cache

    @classmethod
    def from_dynamic_cache(cls, dynamic_cache: DynamicCache, use_ivf: bool = False, index_type: str = "flat"):
        cache = cls()
        key_cache = []
        for k in dynamic_cache.key_cache:
            #key_db = cls.create_key_database(k, flat=(not use_ivf))
            key_db = cls.create_key_database_new(k, index_type=index_type)
            key_cache.append((key_db, torch.empty(0).cuda().to(k.dtype)))
        cache.key_cache = key_cache

        value_cache = []
        for v in dynamic_cache.value_cache:
            value_cache.append((v.cpu(), torch.empty(0).cuda().to(v.dtype)))
        cache.value_cache = value_cache

        # Sequence lengths
        seq_lengths = defaultdict(int)
        for layer in range(len(dynamic_cache.key_cache)):
            seq_lengths[layer] = dynamic_cache.get_seq_length(layer)
        cache.seq_lengths = seq_lengths

        return cache
