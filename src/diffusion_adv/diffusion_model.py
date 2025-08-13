import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return x + self.layers(x)

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, time_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm_weights = nn.LayerNorm(dim)
        self.norm_time = nn.LayerNorm(time_dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(time_dim, dim, bias=False)
        self.to_v = nn.Linear(time_dim, dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, weights, time_emb):
        batch_size = weights.shape[0]
        weights = self.norm_weights(weights)
        time_emb = self.norm_time(time_emb)
        q = self.to_q(weights)
        k = self.to_k(time_emb)
        v = self.to_v(time_emb)
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(batch_size, -1)
        return self.to_out(out)

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim, time_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.time_proj = nn.Linear(time_dim, dim * 2)
    def forward(self, x, time_emb):
        normed = self.norm(x)
        time_params = self.time_proj(time_emb)
        scale, shift = time_params.chunk(2, dim=-1)
        return normed * (1 + scale) + shift

class AdvancedWeightSpaceDiffusion(nn.Module):
    def __init__(self,
                 target_model_flat_dim,
                 time_emb_dim=256,
                 hidden_dim=1024,
                 num_layers=6,
                 num_heads=8,
                 dropout=0.1,
                 use_cross_attention=True,
                 use_adaptive_norm=True):
        super().__init__()
        self.target_model_flat_dim = target_model_flat_dim
        self.time_emb_dim = time_emb_dim
        self.hidden_dim = hidden_dim
        self.use_cross_attention = use_cross_attention
        self.use_adaptive_norm = use_adaptive_norm
        self.time_embedding = SinusoidalPositionEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.input_proj = nn.Linear(target_model_flat_dim, hidden_dim)
        if use_cross_attention:
            self.cross_attention = CrossAttentionBlock(
                hidden_dim, time_emb_dim, num_heads, dropout
            )
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if use_adaptive_norm:
                layer = nn.ModuleList([
                    AdaptiveLayerNorm(hidden_dim, time_emb_dim),
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout)
                ])
            else:
                layer = ResidualBlock(hidden_dim, dropout)
            self.layers.append(layer)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, target_model_flat_dim)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    def forward(self, noisy_weights_flat, t):
        if t.ndim == 2:
            t = t.squeeze(-1)
        time_emb = self.time_embedding(t)
        time_emb = self.time_mlp(time_emb)
        x = self.input_proj(noisy_weights_flat)
        if self.use_cross_attention:
            attn_out = self.cross_attention(x, time_emb)
            x = x + attn_out
        for layer in self.layers:
            if self.use_adaptive_norm:
                norm_layer, linear1, activation, dropout1, linear2, dropout2 = layer
                residual = x
                x = norm_layer(x, time_emb)
                x = linear1(x)
                x = activation(x)
                x = dropout1(x)
                x = linear2(x)
                x = dropout2(x)
                x = x + residual
            else:
                x = layer(x)
        x = self.output_norm(x)
        predicted_denoised_weights_flat = self.output_proj(x)
        return predicted_denoised_weights_flat

def get_target_model_flat_dim(target_model_state_dict):
    return sum(p.numel() for p in target_model_state_dict.values())

def flatten_state_dict(state_dict):
    return torch.cat([p.flatten() for p in state_dict.values()])

def unflatten_to_state_dict(flat_params, reference_state_dict, ks=None):
    new_state_dict = {}
    current_pos = 0
    if ks is None:
        ks = list(reference_state_dict.keys())
    for k in ks:
        param_ref = reference_state_dict[k]
        num_elements = param_ref.numel()
        new_state_dict[k] = flat_params[current_pos : current_pos + num_elements].view(param_ref.shape)
        current_pos += num_elements
    if current_pos != flat_params.numel():
        raise ValueError("Mismatch in number of elements during unflattening.")
    return new_state_dict
# KEY
# SinusoidalPositionEmbedding: time embedding
# ResidualBlock: residual block
# CrossAttentionBlock: cross attention block
# AdaptiveLayerNorm: adaptive layer norm
# AdvancedWeightSpaceDiffusion: diffusion model
# get_target_model_flat_dim: parameter counter
# flatten_state_dict: flattener
# unflatten_to_state_dict: unflattener
