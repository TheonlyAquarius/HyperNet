import torch

def flat(d, ks=None):
    r = []
    if ks is None:
        ks = list(d.keys())
    for k in ks:
        r.append(d[k].reshape(-1))
    return torch.cat(r)

# KEY
# flat: flattener

