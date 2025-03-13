import os
import pytest
import torch
from topk_decoding import AutoTopkModelForCausalLM, TopkCache
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


@pytest.fixture(autouse=True)
def environment():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)


@pytest.fixture
def model_str():
    return "gradientai/Llama-3-8B-Instruct-1048k"


@pytest.fixture
def tokenizer(model_str):
    return AutoTokenizer.from_pretrained(model_str)


@pytest.fixture
def context():
    context = (
        "In a lush, green forest, where the trees whispered secrets to the wind and the rivers sang "
        "melodies of old, lived a clever fox named Felix and a wise turtle named Tessa. Felix was known "
        "far and wide for his cunning ways, while Tessa was respected for her patience and wisdom. "
        "One sunny morning, Felix was trotting through the forest, his mind buzzing with schemes, when "
        "he stumbled upon Tessa slowly making her way across the path. "
    )
    return context


@pytest.fixture
def context_inputs(context, tokenizer):
    context_inputs = tokenizer(context, return_tensors="pt")
    return context_inputs


@pytest.fixture
def prompt():
    return "What happened next?"


@pytest.fixture
def context_prompt_inputs(context, prompt, tokenizer):
    context_prompt_inputs = tokenizer(context + prompt, return_tensors="pt")
    return context_prompt_inputs


@pytest.fixture
def model(model_str):
    model = AutoModelForCausalLM.from_pretrained(
        model_str,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    model = model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    yield model
    model.cpu()
    del model
    torch.cuda.empty_cache()


@pytest.fixture
def model_with_split_mlp(model_str):
    model = AutoTopkModelForCausalLM.from_pretrained(
        model_str,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        mlp=True,
        attn=False,
    ).to("cuda")
    model = model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    yield model
    model.cpu()
    del model
    torch.cuda.empty_cache()


@pytest.fixture
def topk_model(model_str):
    model = AutoTopkModelForCausalLM.from_pretrained(
        model_str,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    model = model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    yield model
    model.cpu()
    del model
    torch.cuda.empty_cache()


@pytest.fixture
def topk_model_with_split_mlp(model_str):
    model = AutoTopkModelForCausalLM.from_pretrained(
        model_str, torch_dtype=torch.bfloat16, mlp=True
    ).to("cuda")
    model = model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    yield model
    model.cpu()
    del model
    torch.cuda.empty_cache()


@pytest.fixture
def topk_model_from_base(model_str):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_str,
        torch_dtype=torch.bfloat16,
    )
    model = AutoTopkModelForCausalLM.from_pretrained(
        model_str, torch_dtype=torch.bfloat16, model=base_model
    ).to("cuda")
    model = model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    yield model
    model.cpu()
    del model
    del base_model
    torch.cuda.empty_cache()


@pytest.fixture
def true_cache(
    context_inputs,
    model,
):
    context_inputs = context_inputs.to(model.device)
    dynamic_cache = DynamicCache()
    dynamic_cache = model(
        **context_inputs, past_key_values=dynamic_cache
    ).past_key_values
    return dynamic_cache


@pytest.fixture
def split_mlp_cache(
    context_inputs,
    model_with_split_mlp,
):
    context_inputs = context_inputs.to(model_with_split_mlp.device)
    dynamic_cache = DynamicCache()
    dynamic_cache = model_with_split_mlp.generate(
        **context_inputs,
        max_new_tokens=1,
        do_sample=False,
        num_beams=1,
        return_dict_in_generate=True,
    ).past_key_values
    return dynamic_cache


@pytest.fixture
def true_outputs(context_prompt_inputs, model, true_cache):
    context_prompt_inputs = context_prompt_inputs.to(model.device)
    with torch.no_grad():
        outputs_true = model.generate(
            **context_prompt_inputs,
            max_new_tokens=5,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            past_key_values=true_cache,
        )
    return outputs_true


@pytest.fixture(params=["topk_model", "topk_model_from_base"])
def topk_outputs_true_cache(request, context_inputs, context_prompt_inputs, true_cache):
    topk_model = request.getfixturevalue(request.param)
    context_prompt_inputs = context_prompt_inputs.to(topk_model.device)
    topk_cache = TopkCache.from_dynamic_cache(true_cache.to("cpu"))
    with torch.no_grad():
        outputs_topk = topk_model.generate(
            **context_prompt_inputs,
            max_new_tokens=5,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            past_key_values=topk_cache,
            k=context_inputs.input_ids.shape[-1],
        )
    return outputs_topk


@pytest.fixture
def topk_outputs_split_mlp_cache(
    context_inputs, context_prompt_inputs, topk_model, split_mlp_cache
):
    context_prompt_inputs = context_prompt_inputs.to(topk_model.device)
    topk_cache = TopkCache.from_dynamic_cache(split_mlp_cache.to("cpu"))
    with torch.no_grad():
        outputs_topk = topk_model.generate(
            **context_prompt_inputs,
            max_new_tokens=5,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            past_key_values=topk_cache,
            k=context_inputs.input_ids.shape[-1],
        )
    return outputs_topk


def test_topk_generate_true_cache(topk_outputs_true_cache, true_outputs, tokenizer):
    true_str = tokenizer.decode(true_outputs.sequences[0])
    topk_str = tokenizer.decode(topk_outputs_true_cache.sequences[0])

    debug_str = "\nEXPECTED OUTPUT:\n" + true_str + "\nTOPK OUTPUT:\n" + topk_str
    assert torch.equal(true_outputs[0], topk_outputs_true_cache[0]), debug_str


def test_topk_generate_split_mlp_cache(
    topk_outputs_split_mlp_cache, true_outputs, tokenizer
):
    true_str = tokenizer.decode(true_outputs.sequences[0])
    topk_str = tokenizer.decode(topk_outputs_split_mlp_cache.sequences[0])

    debug_str = "\nEXPECTED OUTPUT:\n" + true_str + "\nTOPK OUTPUT:\n" + topk_str
    assert torch.equal(true_outputs[0], topk_outputs_split_mlp_cache[0]), debug_str
