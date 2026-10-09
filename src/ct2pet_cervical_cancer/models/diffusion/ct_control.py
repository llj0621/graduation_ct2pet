"""Leaper B1 baseline: CT encoder features control PET latent denoising."""
from dataclasses import dataclass
import torch
import torch.nn as nn
from .components import UNetConfig, UNetEncoder, UNetDecoder, SinusoidalPositionalEmbedding


@dataclass
class CTControlDiffusionConfig(UNetConfig):
    def __post_init__(self):
        self.channel_mult = tuple(self.channel_mult)
        self.attention_levels = tuple(self.attention_levels)
        if self.in_channels != 8 or self.out_channels != 8:
            raise ValueError("CT/PET VAE diffusion requires eight input and output channels")
        if (self.base_channels < 2 or self.time_dim < 2 or self.num_res_blocks < 1
                or self.num_heads < 1 or not self.channel_mult
                or any(value < 1 for value in self.channel_mult)):
            raise ValueError("Invalid U-Net widths, residual block count or time dimension")
        if any(level < 0 or level >= len(self.channel_mult) for level in self.attention_levels):
            raise ValueError("attention_levels must refer to existing U-Net levels")
        if any(self.base_channels * mult % self.num_heads for mult in self.channel_mult):
            raise ValueError("Each channel width must be divisible by num_heads")


class CTControlDiffusion(nn.Module):
    def __init__(self, config: CTControlDiffusionConfig):
        super().__init__()
        self.config = config
        self.time_pos_embed = SinusoidalPositionalEmbedding(config.time_dim)
        self.time_embed = nn.Sequential(nn.Linear(config.time_dim, config.time_dim), nn.SiLU(),
                                        nn.Linear(config.time_dim, config.time_dim))
        self.encoder = UNetEncoder(config)
        self.control_encoder = UNetEncoder(config)
        self.decoder = UNetDecoder(config)

    def forward(self, pet_noisy: torch.Tensor, timesteps: torch.Tensor,
                ct_control: torch.Tensor) -> torch.Tensor:
        if (pet_noisy.ndim != 5 or pet_noisy.shape != ct_control.shape
                or pet_noisy.shape[1] != self.config.in_channels):
            raise ValueError("PET and CT must have identical [B,8,D,H,W] shapes")
        if timesteps.shape != (pet_noisy.shape[0],):
            raise ValueError("Expected one diffusion timestep per sample")
        time_emb = self.time_embed(self.time_pos_embed(timesteps))
        bottleneck, skips = self.encoder(pet_noisy, time_emb)
        control_bottleneck, control_skips = self.control_encoder(ct_control, time_emb)
        combined = [main + control for main, control in zip(skips, control_skips, strict=True)]
        return self.decoder(bottleneck + control_bottleneck, combined, time_emb)
