import torch
import torch.nn as nn
from src.diffusion_adv.diffusion_model import AdvancedWeightSpaceDiffusion


class gen_lin(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.tm1 = nn.Linear(1, n)
        self.lm1 = nn.Linear(n * 2, n)

    def forward(self, w, t):
        e = self.tm1(t)
        x = torch.cat([w, e], 1)
        return self.lm1(x)

# KEY
# model_ref: next weight estimator
# tm1: timestep embedding layer
# lm1: weight mapping layer


def make_gen(n, kind="linear", **kwargs):
    if kind == "linear":
        return gen_lin(n)
    if kind == "diffusion":
        return AdvancedWeightSpaceDiffusion(target_model_flat_dim=n, **kwargs)
    raise ValueError


# KEY
# gen_lin: linear generator
# make_gen: generator factory

