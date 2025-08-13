import importlib
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def train(cfg):
    ds = datasets.MNIST(cfg['dataset'], train=True, download=True, transform=transforms.ToTensor())
    dl = DataLoader(ds, batch_size=cfg['hyperparameters']['batch_size'], shuffle=True)
    m = importlib.import_module(cfg['architecture']).net()
    opt = torch.optim.Adam(m.parameters(), lr=cfg['hyperparameters']['learning_rate'])
    loss = nn.CrossEntropyLoss()
    p = Path(cfg.get('checkpoint_dir', 'checkpoints'))
    p.mkdir(parents=True, exist_ok=True)
    for e in range(cfg['hyperparameters']['epochs']):
        for x, y in dl:
            o = m(x)
            l = loss(o, y)
            opt.zero_grad()
            l.backward()
            opt.step()
        torch.save(m.state_dict(), p / f'{e}.pt')
    return m
