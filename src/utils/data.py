from pathlib import Path
import torch
from torch.utils.data import Dataset
from .flatten import flat

class ckpt_set(Dataset):
    def __init__(self, p):
        self.p = Path(p)
        self.fs = sorted(self.p.glob('*.pt'))
        self.n = len(self.fs) - 1

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        w0 = torch.load(self.fs[i], map_location='cpu')
        w1 = torch.load(self.fs[i + 1], map_location='cpu')
        t = torch.tensor([i / self.n], dtype=torch.float32)
        return flat(w0), flat(w1), t

# KEY
# ckpt_set: dataset of consecutive weight pairs with timestep

