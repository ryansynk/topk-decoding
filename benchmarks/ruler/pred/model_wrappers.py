# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import requests
import os
import sys
import torch
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(PROJECT_ROOT)
from topk_decoding import AutoTopkModelForCausalLM, TopkCache
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache


class HuggingFaceModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=True
        )
        model_kwargs = {"attn_implementation": "sdpa"}
        self.model = AutoTopkModelForCausalLM.from_pretrained(
            name_or_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        )
        self.model_path = name_or_path
        self.generation_kwargs = generation_kwargs
        self.cache = DynamicCache()

        if self.tokenizer.pad_token is None:
            # add pad token to allow batching (known issue for llama2)
            self.tokenizer.padding_side = "left"
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, prompt: str, **kwargs) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(
            self.model.device
        )
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                **self.generation_kwargs,
            )
            generated_text_list = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
        return {"text": generated_text_list}
