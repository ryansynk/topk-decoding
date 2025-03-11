import os
import pytest
import torch
import topk_decoding
from topk_decoding import topk_decoding
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from topk_decoding.topk_model import AutoTopkModelForCausalLM


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


def get_true_outputs(
    context_inputs, prompt_inputs, model_str, tokenizer, return_cache=True
):
    dynamic_cache = DynamicCache()
    model = AutoModelForCausalLM.from_pretrained(
        model_str,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    model = model.eval()
    context_inputs = context_inputs.to(model.device)
    prompt_inputs = prompt_inputs.to(model.device)
    function_outputs = ()
    with torch.no_grad():
        dynamic_cache = model(
            **context_inputs, past_key_values=dynamic_cache
        ).past_key_values
        outputs_true = model.generate(
            **prompt_inputs,
            max_new_tokens=5,
            do_sample=False,
            num_beams=1,
            past_key_values=dynamic_cache,
            return_dict_in_generate=True,
        )
        # Have to re-make dynamic cache because it has tokens from generation added into it
        if return_cache:
            dynamic_cache = DynamicCache()
            dynamic_cache = model(
                **context_inputs, past_key_values=dynamic_cache
            ).past_key_values
            function_outputs = function_outputs + (dynamic_cache,)
        function_outputs = function_outputs + (outputs_true,)

    return function_outputs


def get_topk_outputs(
    context_inputs,
    prompt_inputs,
    context,
    prompt,
    model_str,
    tokenizer,
    dynamic_cache,
    mlp=False,
):
    model = AutoTopkModelForCausalLM.from_pretrained(
        model_str, torch_dtype=torch.bfloat16, mlp=mlp
    ).to("cuda")
    model = model.eval()
    with torch.no_grad():
        topk_kwargs = {}
        topk_kwargs["k"] = context_inputs.input_ids.shape[-1]
        if mlp:
            topk_kwargs["num_mlp_splits"] = 4
        topk_cache = topk_decoding.convert_cache_to_topk(dynamic_cache.to("cpu"))
        context_prompt_inputs = tokenizer(context + prompt, return_tensors="pt").to(
            "cuda"
        )
        prompt_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **context_prompt_inputs,
            max_new_tokens=5,
            do_sample=False,
            num_beams=1,
            past_key_values=topk_cache,
            return_dict_in_generate=True,
            **topk_kwargs,
        )
    return outputs


def test_generate_topk():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    model_str = "gradientai/Llama-3-8B-Instruct-1048k"
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    context = (
        "In a lush, green forest, where the trees whispered secrets to the wind and the rivers sang "
        "melodies of old, lived a clever fox named Felix and a wise turtle named Tessa. Felix was known "
        "far and wide for his cunning ways, while Tessa was respected for her patience and wisdom. "
        "One sunny morning, Felix was trotting through the forest, his mind buzzing with schemes, when "
        "he stumbled upon Tessa slowly making her way across the path. "
    )
    prompt = "What happened next?"
    context_inputs = tokenizer(context, return_tensors="pt")
    prompt_inputs = tokenizer(context + prompt, return_tensors="pt")
    dynamic_cache, outputs_true = get_true_outputs(
        context_inputs, prompt_inputs, model_str, tokenizer
    )
    outputs = get_topk_outputs(
        context_inputs,
        prompt_inputs,
        context,
        prompt,
        model_str,
        tokenizer,
        dynamic_cache,
    )
    output_str_true = tokenizer.decode(outputs_true.sequences[0])
    output_str = tokenizer.decode(outputs.sequences[0])

    debug_str = (
        "\nEXPECTED OUTPUT:\n" + output_str_true + "\nTOPK OUTPUT:\n" + output_str
    )
    assert torch.equal(outputs_true[0], outputs[0]), debug_str


def test_generate_topk_and_unrolled_mlp():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    model_str = "gradientai/Llama-3-8B-Instruct-1048k"
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    context = (
        "In a lush, green forest, where the trees whispered secrets to the wind and the rivers sang "
        "melodies of old, lived a clever fox named Felix and a wise turtle named Tessa. Felix was known "
        "far and wide for his cunning ways, while Tessa was respected for her patience and wisdom. "
        "One sunny morning, Felix was trotting through the forest, his mind buzzing with schemes, when "
        "he stumbled upon Tessa slowly making her way across the path. "
    )
    prompt = "What happened next?"
    context_inputs = tokenizer(context, return_tensors="pt")
    prompt_inputs = tokenizer(context + prompt, return_tensors="pt")
    dynamic_cache, outputs_true = get_true_outputs(
        context_inputs, prompt_inputs, model_str, tokenizer
    )
    outputs = get_topk_outputs(
        context_inputs,
        prompt_inputs,
        context,
        prompt,
        model_str,
        tokenizer,
        dynamic_cache,
        mlp=True,
    )
    output_str_true = tokenizer.decode(outputs_true.sequences[0])
    output_str = tokenizer.decode(outputs.sequences[0])

    debug_str = (
        "\nEXPECTED OUTPUT:\n" + output_str_true + "\nTOPK OUTPUT:\n" + output_str
    )
    assert torch.equal(outputs_true[0], outputs[0]), debug_str
