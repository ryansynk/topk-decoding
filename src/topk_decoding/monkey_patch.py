import inspect
import logging
from topk_decoding.topk_attn import TopkAttention

logger = logging.getLogger(__name__)

def apply_topk_to_llama() -> None:
    from transformers.models.llama import modeling_llama
    from transformers.models.llama.modeling_llama import LlamaModel
    modeling_llama.LlamaAttention = TopkAttention

MODEL_TYPE_TO_APPLY_TOPK_FN = {
    "llama": apply_topk_to_llama
}

def _apply_topk_attn(model_type: str, **kwargs) -> None:
    if not model_type:
        logger.info(f"No model type provided. No topk applied")
        return

    if model_type not in MODEL_TYPE_TO_APPLY_TOPK_FN.keys():
        logger.info(f"There are currently no topk attentions provided for model type: {model_type}.")
        return

    apply_fn = MODEL_TYPE_TO_APPLY_TOPK_FN[model_type]
    apply_fn_signature = inspect.signature(apply_fn)

    applicable_kwargs = {key: value for key, value in kwargs.items() if key in apply_fn_signature.parameters}
    logger.info(f"Applying topk attention to model type: {model_type} with kwargs: {applicable_kwargs}")

    apply_fn(**applicable_kwargs)
