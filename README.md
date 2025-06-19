# topk-decoding
This repo contains the codebase for the paper [Exploiting Sparsity for Long Context Inference](https://arxiv.org/abs/2502.06766)

[!NOTE]
This codebase is currently under construction and the API is subject to large changes.

## RULER
To run RULER, first download the necessary data:
```
cd benchmark/ruler
cd data/synthetic/json/ && python -u download_paulgraham_essay.py && bash download_qa_dataset.sh && cd ../../../
```

Then execute the script:
```
bash run.sh llama-3-8b-1048k ivf 32768 niah_single_1 128 3
```
The parameters of the script are:
-`model_name`
-`index_type`
-`context_length`
-`task`
-`k`
-`num_samples`

## Tests
To run tests, ensure that pytest is installed with `pip install pytest`. Once pytest is installed (and the package itself is installed) simply invoke the command `pytest` from the top-level directory.

## Todo
- Make AutoTopk able to be generated with instantiated model
- Import TopkCache directly, with multiple constructors
- Default k to full context if no k is given with a warning
- Make tests better and more isolated
