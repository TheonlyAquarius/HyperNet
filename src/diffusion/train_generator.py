import torch
from torch import nn
from torch.utils.data import DataLoader
from .generator import model_ref as gen
from src.utils.data import ckpt_set

def train(cfg):
    ds = ckpt_set(cfg['dataset'])
    dl = DataLoader(ds, batch_size=cfg['hyperparameters']['batch_size'], shuffle=True)
    w0, _, _ = ds[0]
    n = w0.numel()
    m = gen(n)
    opt = torch.optim.Adam(m.parameters(), lr=cfg['hyperparameters']['learning_rate'])
    mse = nn.MSELoss()
    for _ in range(cfg['hyperparameters']['epochs']):
        for w0, w1, t in dl:
            o = m(w0, t)
            l = mse(o, w1)
            opt.zero_grad()
            l.backward()
            opt.step()
    return m

# KEY
# train: parameter optimizer for generator using weight pairs and timestep

