import torch
from torch import nn
from transformers.activations import ACT2FN

class UnrolledMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

    def forward(self, x, num_mlp_splits=0):
        if num_mlp_splits > 0:
            chunk_size = x.shape[-2] // num_mlp_splits
            x_slices = x.split(chunk_size, dim=-2)
            down_proj = torch.cat(
                [
                    self.down_proj(self.act_fn(self.gate_proj(x_slice)) * self.up_proj(x_slice))
                    for x_slice in x_slices
                ],
                dim=-2
            )
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj
