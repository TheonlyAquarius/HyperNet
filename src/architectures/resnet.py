import torch
import torch.nn as nn

class blk(nn.Module):
    def __init__(self, f, g, s=1):
        super().__init__()
        self.c1 = nn.Conv2d(f, g, 3, s, 1, bias=False)
        self.b1 = nn.BatchNorm2d(g)
        self.c2 = nn.Conv2d(g, g, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(g)
        self.p = nn.Sequential()
        if s != 1 or f != g:
            self.p = nn.Sequential(nn.Conv2d(f, g, 1, s, bias=False), nn.BatchNorm2d(g))

    def forward(self, x):
        y = self.c1(x)
        y = self.b1(y)
        y = nn.functional.relu(y)
        y = self.c2(y)
        y = self.b2(y)
        y = self.p(x) + y
        return nn.functional.relu(y)

class model_ref(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(1, 16, 3, 1, 1, bias=False)
        self.b = nn.BatchNorm2d(16)
        self.l1 = blk(16, 16)
        self.l2 = blk(16, 32, 2)
        self.l3 = blk(32, 64, 2)
        self.p = nn.AdaptiveAvgPool2d((1, 1))
        self.f = nn.Linear(64, 10)

    def forward(self, x):
        x = self.c(x)
        x = self.b(x)
        x = nn.functional.relu(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.p(x)
        x = torch.flatten(x, 1)
        x = self.f(x)
        return nn.functional.log_softmax(x, 1)

