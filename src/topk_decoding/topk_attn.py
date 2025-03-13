import math
import faiss
import torch
import torch.nn.functional as F
import einops
from torch import nn
from transformers.cache_utils import Cache
from typing import List, Optional, Tuple, Union


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def repeat_kv_db(faiss_cache_db: List[Tuple], n_rep: int) -> List[Tuple]:
    """
    Take each item in the faiss_cache_db and repeat it n_rep times. This is the equivalent of repeat_kv but for lists
    instead of tensors. Used for preparing a KV Cache for multi-query attention.
    """
    new_db = []
    for item in faiss_cache_db:
        for i in range(n_rep):
            new_db.append(item)
    return new_db


class TopkAttention(nn.Module):
    """Topk attention mechanism"""

    def __init__(self, original_attn):
        super().__init__()
        self.topk_k = None
        self.original_attn = original_attn

    @staticmethod
    def get_topk_via_faiss(topk_k, query_states, key_databases, kv_heads, kv_groups):
        """Retrieve top-k values and indices using FAISS.

        Args:
            topk_k (int): Number of top-k values to retrieve.
            query_states (torch.Tensor): Query states tensor of shape (BH, N_q, D).
            key_databases (list): List of FAISS search indexes.
            kv_heads (int): Number of key-value heads.
            kv_groups (int): Number of key-value groups.

        Returns:
            torch.Tensor: Top-k normalized score values tensor of shape (BH, N_q, topk_k).
            torch.Tensor: Top-k indices tensor of shape (BH, N_q, topk_k).
        """
        BH, N_q, D = query_states.shape
        faiss_values_tensor = torch.zeros((BH, N_q, topk_k))
        faiss_indices_tensor = torch.zeros((BH, N_q, topk_k), dtype=torch.int32)
        for i, search_index in enumerate(key_databases):
            faiss_values, faiss_indices = search_index.search(
                query_states[i, :, :].contiguous().to(torch.float32).cpu(), k=topk_k
            )
            faiss_values_tensor[i, :, :] = torch.tensor(
                faiss_values, dtype=faiss_values_tensor.dtype
            )
            faiss_indices_tensor[i, :, :] = torch.tensor(
                faiss_indices, dtype=faiss_indices_tensor.dtype
            )

        # Scale the dot products
        faiss_values_tensor = faiss_values_tensor / math.sqrt(D)
        return faiss_values_tensor, faiss_indices_tensor

    @staticmethod
    def topk_attn(
        topk_k,
        query_states,
        suffix_key_states,
        suffix_value_states,
        prefix_key_db,
        prefix_value_states,
        B,
        H,
        kv_heads,
        kv_groups,
        num_prev_seen_tokens=0,
    ):
        """Computes output of attention block in query phase of model

        In the query phase, the databases from the forward pass in the construct phase have
        already been created. Here we densely compute the attention of the suffix prompt
        with itself, and use the databases to compute the attention of the suffix with
        respect to the prefix.

        Args:
            topk_k (int): number of top-k entries to take
            query_states (torch.Tensor): Query states tensor.
            key_states (torch.Tensor): Key states tensor.
            value_states (torch.Tensor): Value states tensor.
            key_database (list): List of FAISS search indexes.
            value_cache (torch.Tensor): Cached value states.
            B (int): Batch size.
            H (int): Number of heads.

        Returns:
            torch.Tensor: Output tensor of shape (B, H, N_q, D).
        """
        if len(query_states.shape) == 4:
            _, _, N_q, D = query_states.shape
            BH = B * H
            query_states = einops.rearrange(query_states, "B H N D -> (B H) N D")
            prefix_value_states = einops.rearrange(
                prefix_value_states, "B H N D -> (B H) N D"
            )
        else:
            BH, N_q, D = query_states.shape

        N_k_prefix = prefix_key_db[0].ntotal
        if topk_k > N_k_prefix:
            topk_k = N_k_prefix

        topk_values, topk_indices = TopkAttention.get_topk_via_faiss(
            topk_k, query_states, prefix_key_db, kv_heads, kv_groups
        )

        # In query/generation mode we have to combine the prefix keys/values with suffix keys/values and that
        # changes the masking and softmax computation.
        try:
            suffix_value_states = einops.rearrange(
                suffix_value_states, "B H N D -> (B H) N D"
            )
            suffix_key_states = einops.rearrange(
                suffix_key_states, "B H N D -> (B H) N D"
            )
        except (einops.EinopsError, RuntimeError) as e:
            msg = f"Suffix key/value states must not be empty in query/generate mode. Got suffix_key_states: {suffix_key_states.shape if not suffix_key_states is None else type(suffix_key_states)}."
            raise ValueError(msg) from e

        topk_values_exp = torch.exp(topk_values.to(query_states.device))
        topk_values_exp_sums = einops.reduce(
            topk_values_exp, "BH N_q topk_k -> BH N_q", "sum"
        )

        score_dense = query_states @ suffix_key_states.mT / math.sqrt(D)

        # Don't mask if there is only one token
        if N_q > 1:
            score_dense = score_dense + torch.triu(
                torch.full_like(score_dense, float("-inf")), 1
            )
        score_dense_exp = torch.exp(score_dense)
        score_dense_exp_sums = einops.reduce(
            score_dense_exp, "BH N_q N_k -> BH N_q", "sum"
        )
        softmax_denominators = topk_values_exp_sums + score_dense_exp_sums

        attn_dense = score_dense_exp / einops.repeat(
            softmax_denominators,
            "BH N_k_suffix -> BH N_k_suffix N",
            N=score_dense_exp.shape[-1],
        ).to(query_states.dtype)

        attn_sparse = topk_values_exp / einops.repeat(
            softmax_denominators, "BH N_q -> BH N_q topk_k", topk_k=topk_k
        )

        topk_value_states = torch.gather(
            einops.repeat(
                prefix_value_states, "BH N_k_prefix D -> BH N_q N_k_prefix D", N_q=N_q
            ),
            dim=2,
            index=einops.repeat(topk_indices, "BH N_q k -> BH N_q k D", D=D).to(
                torch.int64
            ),
        ).to(attn_dense.device)

        xhat = einops.einsum(
            attn_sparse.to(topk_value_states.dtype),
            topk_value_states,
            "BH N_q k, BH N_q k D -> BH N_q D",
        )

        xhat = xhat + attn_dense @ suffix_value_states
        xhat = einops.rearrange(xhat, "(b h) n d -> b h n d", b=B, h=H)

        return xhat

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        assert self.topk_k is not None, "Topk value not set!"
        topk_k = self.topk_k
        bsz, q_len, _ = hidden_states.size()
        assert not output_attentions, "attentions not generated using faiss"
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.original_attn.head_dim)

        query_states = (
            self.original_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )
        key_states = (
            self.original_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )
        value_states = (
            self.original_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )

        _, self.original_attn.num_heads, _, _ = query_states.shape
        _, self.original_attn.num_key_value_heads, _, _ = key_states.shape

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = position_embeddings
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # "prefix" elements are key or value tensors that are computed offline in a "construct" phase.
        # "suffix" elements are keys/values produced at generation time, or from a "query" that is appended to the text
        # that was used earlier in the offline phase.
        # If update() is called with construct_mode=True, then key/value states are cached as prefix elements for the first
        # time and suffix elements are empty tensors that will not be used.
        # If update() is called with construct_mode=False, then prefix elements are retrieved from an earlier call and suffix
        # elements are updated with each token gnerated.
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
        }
        num_prev_seen_tokens = past_key_value.get_seq_length(
            self.original_attn.layer_idx
        )
        (prefix_key_db, suffix_key_states), (
            prefix_value_states,
            suffix_value_states,
        ) = past_key_value.update(
            key_states, value_states, self.original_attn.layer_idx, cache_kwargs
        )
        if prefix_value_states.ndim == 3:
            prefix_value_states = einops.rearrange(
                prefix_value_states, "n h d -> 1 n h d"
            )
        prefix_key_db = repeat_kv_db(
            prefix_key_db, self.original_attn.num_key_value_groups
        )
        prefix_value_states = repeat_kv(
            prefix_value_states, self.original_attn.num_key_value_groups
        )
        suffix_key_states = repeat_kv(
            suffix_key_states, self.original_attn.num_key_value_groups
        )
        suffix_value_states = repeat_kv(
            suffix_value_states, self.original_attn.num_key_value_groups
        )

        # Note that suffix key/value states are empty tensors that go unused if construct_mode=True
        device = query_states.device
        attn_output = TopkAttention.topk_attn(
            topk_k,
            query_states,
            suffix_key_states,
            suffix_value_states,
            prefix_key_db,
            prefix_value_states,
            bsz,
            self.original_attn.num_heads,
            self.original_attn.num_key_value_heads,
            self.original_attn.num_key_value_groups,
            num_prev_seen_tokens=num_prev_seen_tokens,
        )
        attn_weights = None

        attn_output = attn_output.to(device)
        attn_output = einops.rearrange(attn_output, "B H N D -> B N H D")
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.original_attn.o_proj(attn_output)

        return attn_output, attn_weights
