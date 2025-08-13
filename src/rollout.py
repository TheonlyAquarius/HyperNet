import importlib
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def eval(cfg):
    ds = datasets.MNIST(cfg['dataset'], train=False, download=True, transform=transforms.ToTensor())
    dl = DataLoader(ds, batch_size=cfg['hyperparameters']['batch_size'], shuffle=False)
    m = importlib.import_module(cfg['architecture']).net()
    w = torch.load(cfg['weights'], map_location='cpu')
    m.load_state_dict(w)
    loss = nn.CrossEntropyLoss()
    n = 0
    s = 0.0
    c = 0
    with torch.no_grad():
        for x, y in dl:
            o = m(x)
            l = loss(o, y)
            s += l.item() * x.size(0)
            c += (o.argmax(1) == y).sum().item()
            n += x.size(0)
    return {'loss': s / n, 'acc': c / n}
