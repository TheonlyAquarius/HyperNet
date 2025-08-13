import torch
from torch import nn
from torch.utils.data import DataLoader
from .generator import model_ref as gen
from src.utils.data import ckpt_set

def train(cfg):
    ds = ckpt_set(cfg['dataset'])
    dl = DataLoader(ds, batch_size=cfg['hyperparameters']['batch_size'], shuffle=True)
    n = ds[0].numel()
    m = gen(n)
    opt = torch.optim.Adam(m.parameters(), lr=cfg['hyperparameters']['learning_rate'])
    loss = nn.MSELoss()
    for _ in range(cfg['hyperparameters']['epochs']):
        for d in dl:
            o = m(d)
            l = loss(o, d)
            opt.zero_grad()
            l.backward()
            opt.step()
    return m

