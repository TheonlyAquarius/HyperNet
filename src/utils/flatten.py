import torch

def flat(d):
    r = []
    for k in sorted(d):
        r.append(d[k].reshape(-1))
    return torch.cat(r)

