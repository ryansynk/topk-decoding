import inspect
import logging
import types
from topk_decoding.topk_attn import TopkAttention
from topk_decoding.unrolled_mlp import UnrolledMLP
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def apply_topk_to_llama(
    model: PreTrainedModel, attn: bool = True, mlp: bool = False
) -> None:
    from transformers.models.llama.modeling_llama import LlamaModel

    base_model: LlamaModel = getattr(model, model.base_model_prefix, model)

    if attn:
        for layer in base_model.layers:
            layer.self_attn = TopkAttention(layer.self_attn)
    if mlp:
        for layer in base_model.layers:
            layer.mlp = UnrolledMLP(layer.mlp)


MODEL_TYPE_TO_APPLY_TOPK_FN = {"llama": apply_topk_to_llama}


def _topk_generate(self, *args, **kwargs):
    k = kwargs.pop("k", None)
    for idx, layer in enumerate(self.model.layers):
        if type(layer.self_attn) == TopkAttention:
            assert k is not None, "k not given!"
            layer.self_attn.topk_k = k

    num_mlp_splits = kwargs.pop("num_mlp_splits", None)
    if num_mlp_splits is not None:
        for idx, layer in enumerate(self.model.layers):
            layer.mlp.num_mlp_splits = num_mlp_splits

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
