import torch
import torch.nn as nn
from typing import Dict
from .components import Encoder, Decoder, Codebook as BaseCodebook

__all__ = (
    "PatchVolumn8XEncoder",
    "PatchVolumn4XEncoder",
    "PatchVolumn8XDecoder",
    "PatchVolumn4XDecoder",
    "SemanticCodebook",
)


class SemanticCodebook(nn.Module):
    def __init__(
        self,
        n_codes: int = 16384,
        embedding_dim: int = 8,
        assignment_only_l2: bool = False,
        spherical_vq: bool = False,
    ):
        super().__init__()
        self.codebook = BaseCodebook(
            n_codes,
            embedding_dim,
            assignment_only_l2=assignment_only_l2,
            spherical_vq=spherical_vq,
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.codebook(z)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        n_codes: int,
        embedding_dim: int,
        assignment_only_l2: bool = False,
        spherical_vq: bool = False,
    ):
        model = cls(
            n_codes=n_codes,
            embedding_dim=embedding_dim,
            assignment_only_l2=assignment_only_l2,
            spherical_vq=spherical_vq,
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        cb_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("codebook.codebook."):
                cb_state_dict[k.removeprefix("codebook.")] = v
            elif k.startswith("codebook."):
                cb_state_dict[f"codebook.{k[9:]}"] = v
            elif k.startswith("quantizer."):
                cb_state_dict[f"codebook.{k[10:]}"] = v

        missing, unexpected = model.load_state_dict(cb_state_dict, strict=True)
        # A restored EMA codebook must not be replaced on its first training batch.
        model.codebook._need_init = False
        from loguru import logger

        logger.info(f"Loaded SemanticCodebook from {checkpoint_path}")
        if missing:
            logger.warning(f"Codebook Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Codebook Unexpected keys: {unexpected}")
        return model


class PatchVolumn8XEncoder(nn.Module):
    def __init__(
        self,
        n_hiddens: int = 32,
        image_channels: int = 1,
        norm_type: str = "group",
        num_groups: int = 32,
        embedding_dim: int = 8,
    ):
        super().__init__()
        self.encoder = Encoder(
            n_hiddens=n_hiddens,
            downsample=(8, 8, 8),
            image_channel=image_channels,
            norm_type=norm_type,
            num_groups=num_groups,
            embedding_dim=embedding_dim,
        )
        self.pre_vq_conv = nn.Conv3d(embedding_dim, embedding_dim, 1, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)
        z = self.pre_vq_conv(h)
        return {"embeddings": z}


class PatchVolumn4XEncoder(nn.Module):
    def __init__(
        self,
        n_hiddens: int = 32,
        image_channels: int = 1,
        norm_type: str = "group",
        num_groups: int = 32,
        embedding_dim: int = 8,
    ):
        super().__init__()
        self.encoder = Encoder(
            n_hiddens=n_hiddens,
            downsample=(4, 4, 4),
            image_channel=image_channels,
            norm_type=norm_type,
            num_groups=num_groups,
            embedding_dim=embedding_dim,
        )
        self.pre_vq_conv = nn.Conv3d(embedding_dim, embedding_dim, 1, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)
        z = self.pre_vq_conv(h)
        return {"embeddings": z}


class PatchVolumn8XDecoder(nn.Module):
    def __init__(
        self,
        n_hiddens: int = 32,
        image_channels: int = 1,
        norm_type: str = "group",
        num_groups: int = 32,
        embedding_dim: int = 8,
        use_rope: bool = False,
    ):
        super().__init__()
        self.decoder = Decoder(
            n_hiddens=n_hiddens,
            upsample=(8, 8, 8),
            image_channel=image_channels,
            norm_type=norm_type,
            num_groups=num_groups,
            embedding_dim=embedding_dim,
            use_rope=use_rope,
        )
        self.post_vq_conv = nn.Conv3d(embedding_dim, embedding_dim, 1, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.post_vq_conv(z)
        return self.decoder(h)


class PatchVolumn4XDecoder(nn.Module):
    def __init__(
        self,
        n_hiddens: int = 32,
        image_channels: int = 1,
        norm_type: str = "group",
        num_groups: int = 32,
        embedding_dim: int = 8,
        use_rope: bool = False,
    ):
        super().__init__()
        self.decoder = Decoder(
            n_hiddens=n_hiddens,
            upsample=(4, 4, 4),
            image_channel=image_channels,
            norm_type=norm_type,
            num_groups=num_groups,
            embedding_dim=embedding_dim,
            use_rope=use_rope,
        )
        self.post_vq_conv = nn.Conv3d(embedding_dim, embedding_dim, 1, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.post_vq_conv(z)
        return self.decoder(h)
