import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Literal
from loguru import logger
from .patch import (
    PatchVolumn8XEncoder,
    PatchVolumn4XEncoder,
    PatchVolumn8XDecoder,
    PatchVolumn4XDecoder,
)

__all__ = (
    "CTDecoder",
    "CTEncoder",
    "CTEncoderDecoderConfig",
)


@dataclass
class CTEncoderDecoderConfig:
    n_hiddens: int = 48
    image_channels: int = 1
    norm_type: str = "group"
    num_groups: int = 24
    embedding_dim: int = 8
    n_codes: int = 8192
    scale: Literal["4X", "8X"] = "8X"
    assignment_only_l2: bool = False
    spherical_vq: bool = False
    decoder_rope: bool = False


class CTEncoder(nn.Module):
    def __init__(self, config: CTEncoderDecoderConfig):
        super().__init__()
        self.config = config
        if config.scale == "8X":
            logger.info("Using 8X Patch Volume Encoder")
            self.encoder = PatchVolumn8XEncoder(
                n_hiddens=config.n_hiddens,
                image_channels=config.image_channels,
                norm_type=config.norm_type,
                num_groups=config.num_groups,
                embedding_dim=config.embedding_dim,
            )
        else:
            logger.info("Using 4X Patch Volume Encoder")
            self.encoder = PatchVolumn4XEncoder(
                n_hiddens=config.n_hiddens,
                image_channels=config.image_channels,
                norm_type=config.norm_type,
                num_groups=config.num_groups,
                embedding_dim=config.embedding_dim,
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.encoder(x)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, config: CTEncoderDecoderConfig):
        model = cls(config)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Mapping from patchvolumeAE structure to our structure
        # patchvolumeAE has: encoder, pre_vq_conv, codebook
        # Our CTEncoder has: encoder (which has encoder, pre_vq_conv)

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("encoder.encoder."):
                new_state_dict[k.removeprefix("encoder.")] = v
            elif k.startswith("encoder."):
                new_state_dict[f"encoder.encoder.{k[8:]}"] = v
            elif k.startswith("pre_vq_conv."):
                new_state_dict[f"encoder.pre_vq_conv.{k[12:]}"] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
        logger.info(f"Loaded CTEncoder from {checkpoint_path}")
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
        return model


class CTDecoder(nn.Module):
    def __init__(self, config: CTEncoderDecoderConfig):
        super().__init__()
        self.config = config
        if config.scale == "8X":
            logger.info("Using 8X Patch Volume Decoder")
            self.decoder = PatchVolumn8XDecoder(
                n_hiddens=config.n_hiddens,
                image_channels=config.image_channels,
                norm_type=config.norm_type,
                num_groups=config.num_groups,
                embedding_dim=config.embedding_dim,
                use_rope=config.decoder_rope,
            )
        else:
            logger.info("Using 4X Patch Volume Decoder")
            self.decoder = PatchVolumn4XDecoder(
                n_hiddens=config.n_hiddens,
                image_channels=config.image_channels,
                norm_type=config.norm_type,
                num_groups=config.num_groups,
                embedding_dim=config.embedding_dim,
                use_rope=config.decoder_rope,
            )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, config: CTEncoderDecoderConfig):
        model = cls(config)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("decoder.decoder."):
                new_state_dict[k.removeprefix("decoder.")] = v
            elif k.startswith("decoder."):
                new_state_dict[f"decoder.decoder.{k[8:]}"] = v
            elif k.startswith("post_vq_conv."):
                new_state_dict[f"decoder.post_vq_conv.{k[13:]}"] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=True)
        logger.info(f"Loaded CTDecoder from {checkpoint_path}")
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
        return model
