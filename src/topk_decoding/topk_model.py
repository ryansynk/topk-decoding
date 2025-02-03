import inspect

from topk_decoding.monkey_patch import MODEL_TYPE_TO_APPLY_TOPK_FN, _apply_topk_attn
from transformers import AutoConfig, AutoModelForCausalLM

def _get_model_config(model_dir, **model_init_kwargs):
    config = AutoConfig.from_pretrained(model_dir, **model_init_kwargs)
    return config

class AutoTopkModelForCausalLM(AutoModelForCausalLM):
    """
    This class is a drop-in replacement for AutoModelForCausalLM that applies the Liger Kernel to the model
    if applicable.
    """
    def generate(self, *args, **kwargs):
        k = kwargs.pop("k")
        for idx, layer in enumerate(self.model.layers):
            layer.self_attn.topk_k = k
        super().generate(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model_config = _get_model_config(pretrained_model_name_or_path, **kwargs)

        # Determine the model type and apply the Liger Kernel if applicable
        # Note: _apply_liger_kernel will only pass relevant kwargs to the apply_liger_kernel_to_* function
        model_type = model_config.model_type

        _apply_topk_attn(model_type, **kwargs)

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
