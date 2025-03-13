import functools
import inspect

from transformers import AutoConfig, AutoModelForCausalLM
from topk_decoding.monkey_patch import MODEL_TYPE_TO_APPLY_TOPK_FN, _apply_topk_attn
from topk_decoding.topk_attn import TopkAttention


def _get_model_config(model_dir, **model_init_kwargs):
    config = AutoConfig.from_pretrained(model_dir, **model_init_kwargs)
    return config


class AutoTopkModelForCausalLM(AutoModelForCausalLM):
    """
    This class is a drop-in replacement for AutoModelForCausalLM that applies topk attn to the model
    if applicable.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model_config = _get_model_config(pretrained_model_name_or_path, **kwargs)
        # Determine the model type and apply topk attention if applicable
        # Note: _apply_topk_attn will only pass relevant kwargs to the _apply_topk_kernel_to_* function
        model_type = model_config.model_type

        apply_fn = MODEL_TYPE_TO_APPLY_TOPK_FN[model_type]
        apply_fn_signature = inspect.signature(apply_fn)
        applicable_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in apply_fn_signature.parameters
        }
        base_model = kwargs.pop("model", None)
        if base_model is None:
            model = super().from_pretrained(
                pretrained_model_name_or_path, *model_args, **applicable_kwargs
            )
        else:
            model = base_model

        _apply_topk_attn(model, model_type, **kwargs)

        return model
