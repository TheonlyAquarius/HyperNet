import torch
from torch import nn
from torch.utils.data import DataLoader
from .generator import make_gen
from src.utils.data import ckpt_set


def train(cfg):
    """
    Trains the weight-generating model.
    """
    # Load the dataset of weight transitions (W_t, W_{t+1}, t)
    ds = ckpt_set(cfg['dataset'])
    dl = DataLoader(ds, batch_size=cfg['hyperparameters']['batch_size'], shuffle=True)

    # Determine the flattened weight dimension from the first data point
    w0, _, _ = ds[0]
    n = w0.numel()

    # Create a configurable generator model
    g_cfg = cfg.get('generator', {})
    m = make_gen(n, g_cfg.get('type', 'linear'), **g_cfg.get('args', {}))

    # Set up the optimizer and loss function
    opt = torch.optim.Adam(m.parameters(), lr=cfg['hyperparameters']['learning_rate'])
    mse = nn.MSELoss()

    # Training loop
    for _ in range(cfg['hyperparameters']['epochs']):
        for w0, w1, t in dl:
            # Predict the next weights (W_{t+1}) from the current weights (W_t)
            o = m(w0, t)
            
            # Calculate the loss between the prediction and the actual next weights
            l = mse(o, w1)
            
            # Backpropagation
            opt.zero_grad()
            l.backward()
            opt.step()
            
    return m

# KEY
# train: parameter optimizer for generator using weight pairs and timestep