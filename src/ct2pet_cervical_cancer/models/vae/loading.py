"""Shared model configuration and checkpoint loading for inference."""
from dataclasses import fields
from pathlib import Path
from typing import Any
import torch
from .ct import CTEncoder, CTDecoder, CTEncoderDecoderConfig
from .pet import PETEncoder, PETDecoder, PETEncoderDecoderConfig
from .patch import SemanticCodebook


def load_config(checkpoint_path: Path, config_cls: type) -> Any:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    values = payload.get("hyper_parameters", {})
    return config_cls(**{field.name: values[field.name] for field in fields(config_cls) if field.name in values})


def _classes(modality: str):
    if modality == "ct":
        return CTEncoderDecoderConfig, CTEncoder, CTDecoder
    if modality == "pet":
        return PETEncoderDecoderConfig, PETEncoder, PETDecoder
    raise ValueError(f"Expected 'ct' or 'pet', got {modality!r}")


def load_encoder(checkpoint_path: Path, modality: str, device: torch.device):
    config_cls, encoder_cls, _ = _classes(modality)
    config = load_config(checkpoint_path, config_cls)
    encoder = encoder_cls.load_from_checkpoint(str(checkpoint_path), config).eval().to(device)
    return config, encoder


def _load_codebook(checkpoint_path: Path, config, device: torch.device):
    return SemanticCodebook.load_from_checkpoint(
        str(checkpoint_path), n_codes=config.n_codes, embedding_dim=config.embedding_dim,
        assignment_only_l2=config.assignment_only_l2, spherical_vq=config.spherical_vq,
    ).eval().to(device)


def load_pet_decoder(checkpoint_path: Path, device: torch.device):
    config = load_config(checkpoint_path, PETEncoderDecoderConfig)
    decoder = PETDecoder.load_from_checkpoint(str(checkpoint_path), config).eval().to(device)
    return config, decoder, _load_codebook(checkpoint_path, config, device)


def load_ct_decoder(checkpoint_path: Path, device: torch.device):
    """Load only the CT decoder and its VQ codebook for latent reconstruction."""
    config = load_config(checkpoint_path, CTEncoderDecoderConfig)
    decoder = CTDecoder.load_from_checkpoint(str(checkpoint_path), config).eval().to(device)
    return config, decoder, _load_codebook(checkpoint_path, config, device)


def load_pvae(checkpoint_path: Path, modality: str, device: torch.device):
    config_cls, encoder_cls, decoder_cls = _classes(modality)
    config = load_config(checkpoint_path, config_cls)
    encoder = encoder_cls.load_from_checkpoint(str(checkpoint_path), config).eval().to(device)
    decoder = decoder_cls.load_from_checkpoint(str(checkpoint_path), config).eval().to(device)
    return config, encoder, decoder, _load_codebook(checkpoint_path, config, device)
