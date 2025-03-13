import torch
from torch import nn
from transformers.activations import ACT2FN


class UnrolledMLP(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.num_mlp_splits = 0
        self.mlp = mlp

    def forward(self, x):
        big_enough = x.shape[-2] > self.num_mlp_splits
        if self.num_mlp_splits > 0 and big_enough:
            chunk_size = x.shape[-2] // self.num_mlp_splits
            x_slices = x.split(chunk_size, dim=-2)
            down_proj = torch.cat(
                [
                    self.mlp.down_proj(
                        self.mlp.act_fn(self.mlp.gate_proj(x_slice))
                        * self.mlp.up_proj(x_slice)
                    )
                    for x_slice in x_slices
                ],
                dim=-2,
            )
        else:
            down_proj = self.mlp.down_proj(
                self.mlp.act_fn(self.mlp.gate_proj(x)) * self.mlp.up_proj(x)
            )

        return down_proj
