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

"""
Prepare prediction jsonl with field `pred` .
dataset jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
}

prediction jsonl: 
{
    "index" int,
    "input": str,
    "outputs": [str],
    "pred": str,
}
"""

import argparse
import json
import yaml
import os
import sys
import threading
import importlib
import math
import time
import torch
from tqdm import tqdm
from pathlib import Path
import traceback
from utils import load_data



class ServerAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.server_type = values


parser = argparse.ArgumentParser()
# Data
parser.add_argument("--data_dir", type=Path, required=True, help='path to load the dataset jsonl files')
parser.add_argument("--save_dir", type=Path, required=True, help='path to save the prediction jsonl files')
parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
parser.add_argument("--task", type=str, required=True, help='Options: tasks in benchmark')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--chunk_idx", type=int, default=0, help='index of current split chunk')
parser.add_argument("--chunk_amount", type=int, default=1, help='size of split chunk')

# Server
parser.add_argument("--server_host", type=str, default='127.0.0.1')
parser.add_argument("--server_port", type=str, default='5000')
parser.add_argument("--ssh_server", type=str)
parser.add_argument("--ssh_key_path", type=str)
parser.add_argument("--model_name_or_path", type=str, default='gpt-3.5-turbo', 
                    help='supported models from OpenAI or HF (provide a key or a local path to the checkpoint)')

# Inference
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=32)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--stop_words", type=str, default='')
parser.add_argument("--sliding_window_size", type=int)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--k", type=int, default=128)
parser.add_argument("--index_type", type=str, default="flat")

args = parser.parse_args()
args.stop_words = list(filter(None, args.stop_words.split(',')))
args.threads = 1


def get_llm(tokens_to_generate):
    from model_wrappers import HuggingFaceModel
    # These are all generation_kwargs after model name 
    llm = HuggingFaceModel(
        name_or_path=args.model_name_or_path,
        do_sample=args.temperature > 0,
        max_new_tokens=tokens_to_generate,
        k=args.k,
        index_type=args.index_type
    )
    return llm

def get_output(llm, outputs_parallel, idx, index, input_str, outputs, others, truncation, length):
    try:
        pred = llm(prompt=input_str)
    except Exception as e:
        traceback.print_exc()
        raise

    if len(pred['text']) > 0:
        outputs_parallel[idx] = {
            'index': index,
            'pred': pred['text'][0],
            'input': input_str,
            'outputs': outputs,
            'others': others,
            'truncation': truncation,
            'length': length,
        }

def main():
    start_time = time.time()
    
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    
    try:
        sys.path.append(os.path.dirname(curr_folder))
        module = importlib.import_module(f"data.{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f'{args.task} is not found in config_tasks.yaml')
        
    config = tasks_customized.get(args.task)
    config.update(tasks_base[config['task']])

    task_file = args.data_dir / args.task / f'{args.subset}.jsonl'
    
    if args.chunk_amount > 1:
        pred_file = args.save_dir / f'{args.task}-{args.chunk_idx}.jsonl'
    else:
        pred_file = args.save_dir / f'{args.task}.jsonl'
        
    print(f'Predict {args.task} \nfrom {task_file}\nto {pred_file}')
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    if os.path.exists(pred_file):
        pred_index = [sample['index'] for sample in load_data(pred_file)]
        data = [sample for sample in load_data(task_file) if sample['index'] not in pred_index]
    else:
        data = load_data(task_file)

    # Load api
    llm = get_llm(config['tokens_to_generate'])

    outputs_serial = [{} for _ in range(len(data))]
    with open(pred_file, 'at', encoding="utf-8", buffering=1) as fout:
        for idx, data_point in tqdm(enumerate(data), total=len(data)):
            get_output(
                llm=llm,
                outputs_parallel=outputs_serial,
                idx=idx,
                index=data_point['index'],
                input_str=data_point['input'],
                outputs=data_point['outputs'],
                others=data_point.get('others', {}),
                truncation=data_point.get('truncation', -1),
                length=data_point.get('length', -1),
            )
            if len(outputs_serial[idx]) > 0:
                fout.write(json.dumps(outputs_serial[idx]) + '\n')

    print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")


if __name__ == '__main__':
    main()
