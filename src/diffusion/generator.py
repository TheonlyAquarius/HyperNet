import torch.nn as nn
from src.diffusion_adv.diffusion_model import AdvancedWeightSpaceDiffusion


class gen_lin(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.l = nn.Linear(n, n)

    def forward(self, x):
        return self.l(x)


def make_gen(n, kind="linear", **kwargs):
    if kind == "linear":
        return gen_lin(n)
    if kind == "diffusion":
        return AdvancedWeightSpaceDiffusion(target_model_flat_dim=n, **kwargs)
    raise ValueError


# KEY
# gen_lin: linear generator
# make_gen: generator factory

