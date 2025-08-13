import torch
from torch import nn
from torch.utils.data import DataLoader
from .generator import make_gen
from src.utils.data import ckpt_set


def train(cfg):
    ds = ckpt_set(cfg["dataset"])
    dl = DataLoader(ds, batch_size=cfg["hyperparameters"]["batch_size"], shuffle=True)
    s = ds[0]
    n = s[0].numel() if isinstance(s, tuple) else s.numel()
    g_cfg = cfg.get("generator", {})
    m = make_gen(n, g_cfg.get("type", "linear"), **g_cfg.get("args", {}))
    opt = torch.optim.Adam(m.parameters(), lr=cfg["hyperparameters"]["learning_rate"])
    loss = nn.MSELoss()
    for _ in range(cfg["hyperparameters"]["epochs"]):
        for d in dl:
            if isinstance(d, tuple):
                w, t = d
                o = m(w, t)
                l = loss(o, w)
            else:
                o = m(d)
                l = loss(o, d)
            opt.zero_grad()
            l.backward()
            opt.step()
    return m


# KEY
# train: generator trainer

