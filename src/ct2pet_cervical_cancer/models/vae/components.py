import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange

__all__ = (
    "Normalize",
    "SiLU",
    "ResBlockX",
    "ResBlockXY",
    "Upsample",
    "RotaryPositionEmbedding3D",
    "AttentionBlock",
    "Codebook",
    "Encoder",
    "Decoder",
)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return silu(x)


def shift_dim(
    x: torch.Tensor, src_dim: int = -1, dest_dim: int = -1, make_contiguous: bool = True
) -> torch.Tensor:
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x


def Normalize(
    in_channels: int, norm_type: str = "group", num_groups: int = 32
) -> nn.Module:
    assert norm_type in ["group", "batch"]
    if norm_type == "group":
        return torch.nn.GroupNorm(
            num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
    elif norm_type == "batch":
        return torch.nn.SyncBatchNorm(in_channels)


class ResBlockX(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        norm_type: str = "group",
        num_groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=1
        )
        self.norm2 = Normalize(out_channels, norm_type, num_groups=num_groups)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, padding=1, stride=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            # Although original code didn't have this in ResBlockX,
            # usually resblocks need to handle channel mismatch.
            # But PatchVolume uses ResBlockXY for that.
            return h
        return x + h


class ResBlockXY(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        norm_type: str = "group",
        num_groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.resConv = nn.Conv3d(in_channels, out_channels, (1, 1, 1))
        self.norm1 = Normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=1
        )
        self.norm2 = Normalize(out_channels, norm_type, num_groups=num_groups)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, padding=1, stride=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.resConv(x)
        h = self.norm1(x)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv_trans = nn.ConvTranspose3d(in_channels, in_channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_trans(x)


class RotaryPositionEmbedding3D(nn.Module):
    """Apply separable depth, height, and width RoPE to attention heads."""

    def __init__(self, dim_head: int, theta: float = 10000.0):
        super().__init__()
        # Each axis needs pairs of features. Any remainder stays unrotated.
        self.axis_dim = (dim_head // 6) * 2
        if self.axis_dim == 0:
            raise ValueError("dim_head must be at least 6 for 3D RoPE")
        inv_freq = 1.0 / (
            theta
            ** (torch.arange(0, self.axis_dim, 2, dtype=torch.float32) / self.axis_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rotate_axis(
        self, features: torch.Tensor, coordinates: torch.Tensor
    ) -> torch.Tensor:
        angles = coordinates.to(self.inv_freq.dtype).unsqueeze(-1) * self.inv_freq
        cos = angles.cos().to(features.dtype).unsqueeze(0).unsqueeze(0)
        sin = angles.sin().to(features.dtype).unsqueeze(0).unsqueeze(0)

        paired = features.unflatten(-1, (-1, 2))
        first, second = paired.unbind(dim=-1)
        rotated = torch.stack(
            (first * cos - second * sin, first * sin + second * cos), dim=-1
        )
        return rotated.flatten(start_dim=-2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        spatial_shape: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        depth, height, width = spatial_shape
        coordinates = torch.meshgrid(
            torch.arange(depth, device=q.device),
            torch.arange(height, device=q.device),
            torch.arange(width, device=q.device),
            indexing="ij",
        )
        coordinates = [coordinate.flatten() for coordinate in coordinates]

        rotated_q = []
        rotated_k = []
        offset = 0
        for coordinate in coordinates:
            end = offset + self.axis_dim
            rotated_q.append(self._rotate_axis(q[..., offset:end], coordinate))
            rotated_k.append(self._rotate_axis(k[..., offset:end], coordinate))
            offset = end

        rotated_q.append(q[..., offset:])
        rotated_k.append(k[..., offset:])
        return torch.cat(rotated_q, dim=-1), torch.cat(rotated_k, dim=-1)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        norm_type: str = "group",
        num_groups: int = 32,
        use_rope: bool = False,
    ):
        super().__init__()
        self.norm = Normalize(dim, norm_type=norm_type, num_groups=num_groups)
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)
        self.rope = RotaryPositionEmbedding3D(dim_head) if use_rope else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, z, h, w = x.shape
        x_norm = self.norm(x)
        x_norm = rearrange(x_norm, "b c z x y -> b (z x y) c").contiguous()
        qkv = self.to_qkv(x_norm).chunk(3, dim=2)
        q, k, v = [rearrange(t, "b d (h c) -> b h d c", h=self.heads) for t in qkv]
        if self.rope is not None:
            q, k = self.rope(q, k, (z, h, w))
        out = F.scaled_dot_product_attention(
            q, k, v, scale=self.scale, dropout_p=0.0, is_causal=False
        )
        out = rearrange(
            out, "b h (z x y) c -> b (h c) z x y ", z=z, x=h, y=w
        ).contiguous()
        out = self.to_out(out)
        return out + x


class Codebook(nn.Module):
    def __init__(
        self,
        n_codes: int,
        embedding_dim: int,
        no_random_restart: bool = False,
        restart_thres: float = 1.0,
        assignment_only_l2: bool = False,
        spherical_vq: bool = False,
    ):
        super().__init__()
        if assignment_only_l2 and spherical_vq:
            raise ValueError(
                "assignment_only_l2 and spherical_vq are mutually exclusive"
            )

        self.register_buffer("embeddings", torch.randn(n_codes, embedding_dim))
        self.register_buffer("N", torch.zeros(n_codes))
        self.register_buffer("z_avg", self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True
        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres
        self.assignment_only_l2 = assignment_only_l2
        self.spherical_vq = spherical_vq

    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        self._need_init = False
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        y = self._tile(flat_inputs)

        _k_rand = y[torch.randperm(y.shape[0])][: self.n_codes]
        if self.spherical_vq:
            _k_rand = F.normalize(_k_rand, dim=-1)
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))

    def forward(self, z: torch.Tensor) -> dict:
        if self._need_init and self.training:
            self._init_embeddings(z)
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)  # [bthw, c]

        if self.assignment_only_l2 or self.spherical_vq:
            assignment_inputs = F.normalize(flat_inputs, dim=-1)
            assignment_embeddings = F.normalize(self.embeddings, dim=-1)
        else:
            assignment_inputs = flat_inputs
            assignment_embeddings = self.embeddings

        distances = (
            (assignment_inputs**2).sum(dim=1, keepdim=True)
            - 2 * assignment_inputs @ assignment_embeddings.t()
            + (assignment_embeddings.t() ** 2).sum(dim=0, keepdim=True)
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(
            flat_inputs
        )  # [bthw, ncode]

        # Reshape indices to spatial shape [B, D, H, W]
        encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:])

        # F.embedding handles the spatial shape automatically: [B, D, H, W] -> [B, D, H, W, C]
        lookup_embeddings = (
            assignment_embeddings if self.spherical_vq else self.embeddings
        )
        embeddings = F.embedding(encoding_indices, lookup_embeddings)
        embeddings = shift_dim(embeddings, -1, 1)  # [B, C, D, H, W]

        commitment_input = F.normalize(z, dim=1) if self.spherical_vq else z
        commitment_loss = 0.25 * F.mse_loss(commitment_input, embeddings.detach())

        # EMA codebook update
        if self.training:
            n_total = encode_onehot.sum(dim=0)
            ema_inputs = assignment_inputs if self.spherical_vq else flat_inputs
            encode_sum = ema_inputs.t() @ encode_onehot
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            if self.spherical_vq:
                encode_normalized = F.normalize(encode_normalized, dim=-1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(ema_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][: self.n_codes]
            if self.spherical_vq:
                _k_rand = F.normalize(_k_rand, dim=-1)
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            if not self.no_random_restart:
                usage = (self.N.view(self.n_codes, 1) >= self.restart_thres).float()
                self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        embeddings_st = (embeddings - commitment_input).detach() + commitment_input

        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7)))

        return dict(
            embeddings=embeddings_st,
            encodings=encoding_indices,
            commitment_loss=commitment_loss,
            perplexity=perplexity,
        )

    def dictionary_lookup(self, encodings: torch.Tensor) -> torch.Tensor:
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings


class Encoder(nn.Module):
    def __init__(
        self,
        n_hiddens: int,
        downsample: Tuple[int, int, int],
        image_channel: int = 1,
        norm_type: str = "group",
        num_groups: int = 32,
        embedding_dim: int = 8,
    ):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.conv_blocks = nn.ModuleList()
        max_ds = n_times_downsample.max()
        self.embedding_dim = embedding_dim
        self.conv_first = nn.Conv3d(
            image_channel, n_hiddens, kernel_size=3, stride=1, padding=1
        )

        channels = [n_hiddens * 2**i for i in range(max_ds)]
        channels = channels + [channels[-1]]
        in_channels = channels[0]
        for i in range(max_ds + 1):
            block = nn.Module()
            if i != 0:
                in_channels = channels[i - 1]
            out_channels = channels[i]
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            if in_channels != out_channels:
                block.res1 = ResBlockXY(
                    in_channels,
                    out_channels,
                    norm_type=norm_type,
                    num_groups=num_groups,
                )
            else:
                block.res1 = ResBlockX(
                    in_channels,
                    out_channels,
                    norm_type=norm_type,
                    num_groups=num_groups,
                )

            block.res2 = ResBlockX(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups
            )
            if i != max_ds:
                block.down = nn.Conv3d(
                    out_channels,
                    out_channels,
                    kernel_size=(4, 4, 4),
                    stride=stride,
                    padding=1,
                )
            else:
                block.down = nn.Identity()
            self.conv_blocks.append(block)
            n_times_downsample -= 1
        self.mid_block = nn.Module()
        self.mid_block.res1 = ResBlockX(
            out_channels, out_channels, norm_type=norm_type, num_groups=num_groups
        )
        self.mid_block.attn = AttentionBlock(
            out_channels, heads=4, norm_type=norm_type, num_groups=num_groups
        )
        self.mid_block.res2 = ResBlockX(
            out_channels, out_channels, norm_type=norm_type, num_groups=num_groups
        )
        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type, num_groups=num_groups),
            SiLU(),
            nn.Conv3d(out_channels, self.embedding_dim, 3, 1, 1),
        )

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_first(x)
        for idx, block in enumerate(self.conv_blocks):
            h = block.res1(h)
            h = block.res2(h)
            h = block.down(h)
        h = self.mid_block.res1(h)
        h = self.mid_block.attn(h)
        h = self.mid_block.res2(h)
        h = self.final_block(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        n_hiddens: int,
        upsample: Tuple[int, int, int],
        image_channel: int,
        norm_type: str = "group",
        num_groups: int = 32,
        embedding_dim: int = 8,
        use_rope: bool = False,
    ):
        super().__init__()

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        channels = [n_hiddens * 2**i for i in range(max_us)]
        channels = channels + [channels[-1]]
        channels.reverse()
        self.embedding_dim = embedding_dim
        self.conv_first = nn.Conv3d(self.embedding_dim, channels[0], 3, 1, 1)
        self.mid_block = nn.Module()
        self.mid_block.res1 = ResBlockX(
            channels[0], channels[0], norm_type=norm_type, num_groups=num_groups
        )
        self.mid_block.attn = AttentionBlock(
            channels[0],
            heads=4,
            norm_type=norm_type,
            num_groups=num_groups,
            use_rope=use_rope,
        )
        self.mid_block.res2 = ResBlockX(
            channels[0], channels[0], norm_type=norm_type, num_groups=num_groups
        )
        self.conv_blocks = nn.ModuleList()
        in_channels = channels[0]
        for i in range(max_us + 1):
            block = nn.Module()
            if i != 0:
                in_channels = channels[i - 1]
            out_channels = channels[i]
            # us = tuple([2 if d > 0 else 1 for d in n_times_upsample]) # not used in original code?
            if in_channels != out_channels:
                block.res1 = ResBlockXY(
                    in_channels,
                    out_channels,
                    norm_type=norm_type,
                    num_groups=num_groups,
                )
            else:
                block.res1 = ResBlockX(
                    in_channels,
                    out_channels,
                    norm_type=norm_type,
                    num_groups=num_groups,
                )
            block.res2 = ResBlockX(
                out_channels, out_channels, norm_type=norm_type, num_groups=num_groups
            )
            if i != max_us:
                block.up = Upsample(out_channels)
            else:
                block.up = nn.Identity()
            self.conv_blocks.append(block)
            n_times_upsample -= 1

        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type, num_groups=num_groups),
            SiLU(),
            nn.Conv3d(out_channels, image_channel, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_first(x)
        h = self.mid_block.res1(h)
        h = self.mid_block.attn(h)
        h = self.mid_block.res2(h)
        for i, block in enumerate(self.conv_blocks):
            h = block.res1(h)
            h = block.res2(h)
            h = block.up(h)
        h = self.final_block(h)
        return h
