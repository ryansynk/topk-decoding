#!/bin/bash
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

# container: docker.io/cphsieh/ruler:0.1.0
# bash run.sh MODEL_NAME BENCHMARK_NAME

if [ $# -ne 6 ]; then
    echo "Usage: $0 <model_name> $1 <index_type> $2 <context length> $3 <task> $4 <k> $5 <num_samples>"
    exit 1
fi


# Root Directories
# GPUS="1" # GPU size for tensor_parallel.
# MODEL_DIR="../.." # the path that contains individual model folders from HUggingface.
# ENGINE_DIR="." # the path that contains individual engine folders from TensorRT-LLM.
# BATCH_SIZE=1  # increase to improve GPU utilization

ROOT_DIR="./ruler_eval_results" # the path that stores generated task samples and model predictions.
NUM_SAMPLES=${6}
MAX_SEQ_LENGTH=${3}
INDEX_TYPE=${2}
DEVICE=auto


# Model and Tokenizer
source config_models.sh
MODEL_NAME=${1}
MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME})
IFS=":" read MODEL_NAME MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE <<< "$MODEL_CONFIG"
if [ -z "${MODEL_NAME}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi


# Benchmark and Tasks
source config_tasks.sh
BENCHMARK="synthetic"


RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
DATA_DIR="${RESULTS_DIR}/data"
PRED_DIR="${RESULTS_DIR}/pred"
mkdir -p ${DATA_DIR}
mkdir -p ${PRED_DIR}
    
TASK=${4}
total_time=0
python data/prepare.py \
    --save_dir ${DATA_DIR} \
    --benchmark ${BENCHMARK} \
    --task ${TASK} \
    --tokenizer_path ${TOKENIZER_PATH} \
    --tokenizer_type ${TOKENIZER_TYPE} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --model_template_type ${MODEL_TEMPLATE_TYPE} \
    --num_samples ${NUM_SAMPLES} \
    --random_seed 42 \
    ${REMOVE_NEWLINE_TAB}

K=${5}
python pred/call_api.py \
    --model_name_or_path ${MODEL_NAME} \
    --batch_size 1 \
    --data_dir ${DATA_DIR} \
    --save_dir ${PRED_DIR} \
    --benchmark ${BENCHMARK} \
    --task ${TASK} \
    --device ${DEVICE} \
    --index_type ${INDEX_TYPE} \
    --k ${K} \
    ${STOP_WORDS}
    
python eval/evaluate.py \
    --data_dir ${PRED_DIR} \
    --benchmark ${BENCHMARK}
