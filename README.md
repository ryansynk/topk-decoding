# topk-decoding
This repo contains the codebase for the paper [Exploiting Sparsity for Long Context Inference](https://arxiv.org/abs/2502.06766)

[!NOTE]
This codebase is currently under construction and the API is subject to large changes.

## Tests
To run tests, ensure that pytest is installed with `pip install pytest`. Once pytest is installed (and the package itself is installed) simply invoke the command `pytest` from the top-level directory.

## Todo
- Make AutoTopk able to be generated with instantiated model
- Import TopkCache directly, with multiple constructors
- Default k to full context if no k is given with a warning
- Make tests better and more isolated
