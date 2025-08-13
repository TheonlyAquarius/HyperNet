from pathlib import Path
import torch
from torch.utils.data import Dataset
from .flatten import flat

class ckpt_set(Dataset):
    def __init__(self, p):
        self.p = Path(p)
        self.fs = sorted(self.p.glob('*.pt'))

    def __len__(self):
        return len(self.fs)

    def __getitem__(self, i):
        w = torch.load(self.fs[i], map_location='cpu')
        return flat(w)

