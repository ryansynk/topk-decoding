import functools
import types

from transformers import AutoConfig, AutoModelForCausalLM
from topk_decoding.monkey_patch import MODEL_TYPE_TO_APPLY_TOPK_FN, _apply_topk_attn
from topk_decoding.topk_attn import TopkAttention

def _get_model_config(model_dir, **model_init_kwargs):
    config = AutoConfig.from_pretrained(model_dir, **model_init_kwargs)
    return config

def _topk_generate(self, *args, **kwargs):
    k = kwargs.pop("k")
    for idx, layer in enumerate(self.model.layers):
        layer.self_attn.topk_k = k
    return self.original_generate(*args, **kwargs)

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

        _apply_topk_attn(model_type, **kwargs)

        model = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        model.original_generate = model.generate
        model.generate = types.MethodType(_topk_generate, model)
        return model
