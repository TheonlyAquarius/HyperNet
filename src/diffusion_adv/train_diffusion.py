import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import glob
import time
from .diffusion_model import AdvancedWeightSpaceDiffusion, flatten_state_dict, get_target_model_flat_dim
from .target_model import TargetModel

class WeightCheckpointsDataset(Dataset):
    def __init__(self, checkpoints_dir, reference_state_dict, max_checkpoints_len=None):
        self.checkpoints_dir = checkpoints_dir
        self.reference_state_dict = reference_state_dict
        self.flat_dim = get_target_model_flat_dim(reference_state_dict)
        weight_files = sorted(
            glob.glob(os.path.join(checkpoints_dir, "weights_epoch_*.safetensors")),
            key=lambda x: int(x.split('_')[-1].split('.')[0]) if x.split('_')[-1].split('.')[0].isdigit() else -1
        )
        if not weight_files:
            raise FileNotFoundError(f"No weight files found in {checkpoints_dir}.")
        self.weight_files = weight_files
        if max_checkpoints_len:
            self.weight_files = self.weight_files[:max_checkpoints_len + 1]
        self.num_total_steps = len(self.weight_files) - 1
    def __len__(self):
        return self.num_total_steps
    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError(f"Index {idx} out of range")
        w_path_current = self.weight_files[idx]
        w_path_next = self.weight_files[idx+1]
        state_dict_current = torch.load(w_path_current, map_location='cpu')
        current_w = flatten_state_dict(state_dict_current)
        state_dict_next = torch.load(w_path_next, map_location='cpu')
        target_next_w = flatten_state_dict(state_dict_next)
        if current_w.shape[0] != self.flat_dim or target_next_w.shape[0] != self.flat_dim:
            raise ValueError("Dimension mismatch in loaded weights")
        t = torch.tensor([float(idx) / (self.num_total_steps - 1) if self.num_total_steps > 1 else 0.0])
        return current_w, target_next_w, t

def train_diffusion_model(
    checkpoints_dir,
    target_model_reference,
    epochs=100,
    lr=0.001,
    batch_size=128,
    time_emb_dim=256,
    hidden_dim_diff_model=1024,
    num_layers=6,
    num_heads=8,
    dropout=0.1,
    use_cross_attention=True,
    use_adaptive_norm=True,
    save_path="trained_diffusion_model.safetensors",
    max_traj_len_for_training=None,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(target_model_reference, nn.Module):
        reference_state_dict = target_model_reference.state_dict()
    else:
        reference_state_dict = target_model_reference
    target_flat_dim = get_target_model_flat_dim(reference_state_dict)
    diffusion_model = AdvancedWeightSpaceDiffusion(
        target_model_flat_dim=target_flat_dim,
        time_emb_dim=time_emb_dim,
        hidden_dim=hidden_dim_diff_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        use_cross_attention=use_cross_attention,
        use_adaptive_norm=use_adaptive_norm
    ).to(device)
    dataset = WeightCheckpointsDataset(checkpoints_dir, reference_state_dict, max_checkpoints_len=max_traj_len_for_training)
    if len(dataset) == 0:
        return
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(diffusion_model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        diffusion_model.train()
        total_loss = 0
        for batch_idx, (current_weights_flat, target_next_weights_flat, timesteps_t) in enumerate(dataloader):
            current_weights_flat = current_weights_flat.to(device)
            target_next_weights_flat = target_next_weights_flat.to(device)
            timesteps_t = timesteps_t.to(device)
            optimizer.zero_grad()
            predicted_next_weights_flat = diffusion_model(current_weights_flat, timesteps_t)
            loss = criterion(predicted_next_weights_flat, target_next_weights_flat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch}/{epochs}], Average Loss: {avg_loss:.6f} | Time: {time.time()-start_time:.2f}s")
    torch.save(diffusion_model.state_dict(), save_path)

# KEY
# WeightCheckpointsDataset: dataset
# train_diffusion_model: diffusion trainer
