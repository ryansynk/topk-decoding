import os
import pytest
import torch
import topk_decoding
from topk_decoding import topk_decoding
from topk_decoding.topk_model import AutoTopkModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def test_import_convert_cache_to_topk():
    with pytest.raises(NotImplementedError) as e_info:
        topk_decoding.convert_cache_to_topk(None)


def test_import_convert_model_to_topk():
    model = AutoModelForCausalLM.from_pretrained(
        "/fs/nexus-scratch/ryansynk/.cache/huggingface/hub/models--gradientai--Llama-3-8B-Instruct-1048k/snapshots/8697fb25cb77c852311e03b4464b8467471d56a4/",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    model = topk_decoding.convert_model_to_topk(model)
    for layer in model.model.layers:
        assert isinstance(layer.self_attn, topk_decoding.TopkAttention), type(
            layer.self_attn
        )

def test_generate_topk():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    model = AutoModelForCausalLM.from_pretrained(
        "/fs/nexus-scratch/ryansynk/.cache/huggingface/hub/models--gradientai--Llama-3-8B-Instruct-1048k/snapshots/8697fb25cb77c852311e03b4464b8467471d56a4/",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("/fs/nexus-scratch/ryansynk/.cache/huggingface/hub/models--gradientai--Llama-3-8B-Instruct-1048k/snapshots/8697fb25cb77c852311e03b4464b8467471d56a4/")
    context = (
        "In a lush, green forest, where the trees whispered secrets to the wind and the rivers sang "
        "melodies of old, lived a clever fox named Felix and a wise turtle named Tessa. Felix was known "
        "far and wide for his cunning ways, while Tessa was respected for her patience and wisdom. "
        "One sunny morning, Felix was trotting through the forest, his mind buzzing with schemes, when "
        "he stumbled upon Tessa slowly making her way across the path. "
    )
    prompt = "What happened next?"
    context_inputs = tokenizer(context, return_tensors="pt").to(model.device)
    prompt_inputs = tokenizer(context + prompt, return_tensors="pt").to("cuda")
    dynamic_cache = DynamicCache()
    model = model.eval()
    with torch.no_grad():
        dynamic_cache = model(
            **context_inputs, 
            past_key_values=dynamic_cache).past_key_values
        outputs_true = model.generate(
            **prompt_inputs, 
            max_new_tokens=5, 
            do_sample=False,
            num_beams=1
        )

    model = AutoTopkModelForCausalLM.from_pretrained(
        "/fs/nexus-scratch/ryansynk/.cache/huggingface/hub/models--gradientai--Llama-3-8B-Instruct-1048k/snapshots/8697fb25cb77c852311e03b4464b8467471d56a4/",
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    model = model.eval()
    for idx, layer in enumerate(model.model.layers):
        layer.self_attn.topk_k = context_inputs.input_ids.shape[-1]
    with torch.no_grad():
        topk_cache = topk_decoding.convert_cache_to_topk(dynamic_cache.to("cpu"))
        prompt_inputs = tokenizer(context + prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **prompt_inputs, 
            max_new_tokens=5, 
            do_sample=False,
            num_beams=1,
            use_cache=True,
            past_key_values=topk_cache,
        )

    output_str_true = tokenizer.decode(outputs_true[0])
    output_str = tokenizer.decode(outputs[0])
    debug_str = "\nEXPECTED OUTPUT:\n" + output_str_true + "\nTOPK OUTPUT:\n" + output_str
    assert torch.equal(outputs_true[0], outputs[0]), debug_str

