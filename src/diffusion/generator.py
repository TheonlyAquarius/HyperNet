import torch.nn as nn

class model_ref(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.l = nn.Linear(n, n)

    def forward(self, x):
        return self.l(x)

