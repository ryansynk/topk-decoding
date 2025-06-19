import inspect
import logging
import types
from topk_decoding.topk_attn import TopkAttention
from topk_decoding.unrolled_mlp import UnrolledMLP
from topk_decoding.topk_cache import TopkCache
from transformers import PreTrainedModel, DynamicCache

logger = logging.getLogger(__name__)


def apply_topk_to_llama(
    model: PreTrainedModel, attn: bool = True, mlp: bool = False
) -> None:
    from transformers.models.llama.modeling_llama import LlamaModel

    base_model: LlamaModel = getattr(model, model.base_model_prefix, model)

    if attn:
        for layer in base_model.layers:
            layer.orig_self_attn = layer.self_attn
            layer.topk_self_attn = TopkAttention(layer.self_attn)
            layer.self_attn = layer.topk_self_attn
    if mlp:
        for layer in base_model.layers:
            layer.orig_mlp = layer.mlp
            layer.unrolled_mlp = UnrolledMLP(layer.mlp)
            layer.mlp = layer.unrolled_mlp


def revert_topk_to_std_attn_llama(model: PreTrainedModel) -> None:
    from transformers.models.llama.modeling_llama import LlamaModel


MODEL_TYPE_TO_APPLY_TOPK_FN = {"llama": apply_topk_to_llama}


def _topk_generate(self, *args, **kwargs):
    # TODO could have a "cache tensor" file
    k = kwargs.pop("k", None)
    for idx, layer in enumerate(self.model.layers):
        if type(layer.self_attn) == TopkAttention:
            assert k is not None, "k not given!"
            layer.topk_self_attn.topk_k = k

    num_mlp_splits = kwargs.pop("num_mlp_splits", None)
    if num_mlp_splits is not None:
        for idx, layer in enumerate(self.model.layers):
            layer.unrolled_mlp.num_mlp_splits = num_mlp_splits

    # if there's no cache,(or its empty) do prefill
    cache = kwargs.get("past_key_values", None)
    if cache is None or cache.get_seq_length() == 0:
        print("No cache detected. Prefilling")
        # patch normal attn back on
        # base_model = getattr(model, model.base_model_prefix, model)
        base_model = getattr(self, self.base_model_prefix, self)
        for layer in base_model.layers:
            layer.self_attn = layer.orig_self_attn

        # do forward pass
        # TODO Offloading? Static?
        # TODO no_grad?
        input_ids_pf = kwargs.get("input_ids")[..., :-1]
        attention_mask_pf = kwargs.get("attention_mask")[..., :-1]
        cache = DynamicCache()
        cache = self.model(
            input_ids=input_ids_pf,
            attention_mask=attention_mask_pf,
            past_key_values=cache,
        ).past_key_values
        index_type = kwargs.pop("index_type", "flat")
        kwargs["past_key_values"] = TopkCache.from_dynamic_cache(
            cache, index_type=index_type
        )

        # patch topk attention back on
        for layer in base_model.layers:
            layer.self_attn = layer.topk_self_attn
        print("Prefill complete")

    return self.original_generate(*args, **kwargs)


def _apply_topk_attn(model: PreTrainedModel, model_type: str, **kwargs) -> None:
    if not model_type:
        logger.info(f"No model type provided. No topk applied")
        return

    if model_type not in MODEL_TYPE_TO_APPLY_TOPK_FN.keys():
        logger.info(
            f"There are currently no topk attentions provided for model type: {model_type}."
        )
        return

    apply_fn = MODEL_TYPE_TO_APPLY_TOPK_FN[model_type]
    apply_fn_signature = inspect.signature(apply_fn)

    applicable_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in apply_fn_signature.parameters
    }
    logger.info(
        f"Applying topk attention to model type: {model_type} with kwargs: {applicable_kwargs}"
    )

    apply_fn(model, **applicable_kwargs)
    model.original_generate = model.generate
    model.generate = types.MethodType(_topk_generate, model)
