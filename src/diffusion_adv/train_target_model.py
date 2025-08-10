import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from safetensors.torch import save_file
from .target_model import TargetModel

def train_target_model(epochs=5, lr=0.01, batch_size=64, save_dir='checkpoints_weights_cnn', download=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=download, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = TargetModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.NLLLoss()
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        save_file(model.state_dict(), os.path.join(save_dir, f'weights_epoch_{epoch}.safetensors'))

# KEY
# download: dataset flag
# train_target_model: cnn trainer
