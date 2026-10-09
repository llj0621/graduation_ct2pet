"""3D U-Net blocks adapted from Leaper's dgp/arch/component_blocks/unet.py.

Keep its residual blocks, attention and skip layout; CT-only conditioning does
not require text cross-attention or deep-supervision heads.
"""
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(channels: int) -> nn.GroupNorm:
    groups = math.gcd(32, channels)
    # Small smoke-test networks must still have multiple values per group.
    if channels // groups == 1 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, channels)


@dataclass
class UNetConfig:
    in_channels: int = 8
    out_channels: int = 8
    base_channels: int = 64
    channel_mult: tuple[int, ...] = (1, 2, 4, 8)
    num_res_blocks: int = 2
    time_dim: int = 256
    attention_levels: tuple[int, ...] = (2, 3)
    num_heads: int = 4


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, device=timesteps.device).float() / half
        )
        angles = timesteps.reshape(-1, 1).float() * frequencies
        return F.pad(torch.cat((angles.sin(), angles.cos()), dim=-1), (0, self.dim % 2))


class ResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = group_norm(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.shortcut = (nn.Conv3d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())
        self.act = nn.SiLU()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels * 2))

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        scale, shift = self.time_mlp(time_emb).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        return self.conv2(self.act(self.norm2(h))) + self.shortcut(x)


class SelfAttention3D(nn.Module):
    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(channels)
        self.ff_self = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels),
                                     nn.SiLU(), nn.Linear(channels, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        normalized = self.ln(tokens)
        attended, _ = self.mha(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        return (tokens + self.ff_self(tokens)).transpose(1, 2).reshape(shape)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, config: UNetConfig, attention: bool):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResBlock3D(in_channels if index == 0 else out_channels, out_channels, config.time_dim)
            for index in range(config.num_res_blocks)
        ])
        self.self_attn = SelfAttention3D(out_channels, config.num_heads) if attention else None
        self.downsample = nn.Conv3d(out_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        for block in self.res_blocks:
            x = block(x, time_emb)
        if self.self_attn is not None:
            x = self.self_attn(x)
        return self.downsample(x), x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, config: UNetConfig, attention: bool):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels, 2, stride=2)
        self.res_blocks = nn.ModuleList([
            ResBlock3D(in_channels + out_channels if index == 0 else out_channels,
                       out_channels, config.time_dim)
            for index in range(config.num_res_blocks)
        ])
        self.self_attn = SelfAttention3D(out_channels, config.num_heads) if attention else None

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat((x, skip), dim=1)
        for block in self.res_blocks:
            x = block(x, time_emb)
        return self.self_attn(x) if self.self_attn is not None else x


class MiddleBlock(nn.Module):
    def __init__(self, channels: int, config: UNetConfig):
        super().__init__()
        self.res1 = ResBlock3D(channels, channels, config.time_dim)
        self.self_attn = SelfAttention3D(channels, config.num_heads)
        self.res2 = ResBlock3D(channels, channels, config.time_dim)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        return self.res2(self.self_attn(self.res1(x, time_emb)), time_emb)


class UNetEncoder(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        self.conv_in = nn.Conv3d(config.in_channels, config.base_channels, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        channels = config.base_channels
        for level, multiplier in enumerate(config.channel_mult):
            next_channels = config.base_channels * multiplier
            self.down_blocks.append(DownBlock(channels, next_channels, config,
                                               level in config.attention_levels))
            channels = next_channels
        self.middle = MiddleBlock(channels, config)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):
        x = self.conv_in(x)
        skips = []
        for block in self.down_blocks:
            x, skip = block(x, time_emb)
            skips.append(skip)
        return self.middle(x, time_emb), skips


class UNetDecoder(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        channels = config.base_channels * config.channel_mult[-1]
        self.up_blocks = nn.ModuleList()
        for level in reversed(range(len(config.channel_mult))):
            next_channels = config.base_channels * config.channel_mult[level]
            self.up_blocks.append(UpBlock(channels, next_channels, config,
                                           level in config.attention_levels))
            channels = next_channels
        self.norm_out = group_norm(channels)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv3d(channels, config.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor], time_emb: torch.Tensor):
        for block, skip in zip(self.up_blocks, reversed(skips), strict=True):
            x = block(x, skip, time_emb)
        return self.conv_out(self.act_out(self.norm_out(x)))
