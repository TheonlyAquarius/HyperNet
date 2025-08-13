import torch
import torch.nn as nn

class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, 1)
        self.c2 = nn.Conv2d(32, 64, 3, 1)
        self.d1 = nn.Dropout(0.25)
        self.d2 = nn.Dropout(0.5)
        self.f1 = nn.Linear(9216, 128)
        self.f2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.c1(x)
        x = nn.functional.relu(x)
        x = self.c2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.d1(x)
        x = torch.flatten(x, 1)
        x = self.f1(x)
        x = nn.functional.relu(x)
        x = self.d2(x)
        x = self.f2(x)
        return x
