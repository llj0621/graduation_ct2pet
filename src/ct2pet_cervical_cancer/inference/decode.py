import torch
import torch.nn as nn
from typing import List, Literal, Optional, Sequence, Tuple, Union
from loguru import logger
from .encode import get_gaussian_window, get_sliding_window_starts
from ..models.vae.patch import SemanticCodebook
from ..datasets.latent_transform import LatentTransform

__all__ = ("decode_sliding", "longitudinal_sample")


def longitudinal_sample(
    volume: torch.Tensor,
    slices: Sequence[float] = (0.25, 0.5, 0.75),
    axis: Literal["sagittal", "coronal", "axial"] = "sagittal",
) -> List[torch.Tensor]:
    """Sample planes at fractional positions along a spatial axis.

    Inputs are expected in RAS tensor order ``[B, C, X, Y, Z]`` or
    ``[C, X, Y, Z]``. The default sagittal plane fixes X and retains Y-Z,
    which is a true longitudinal view. Returned tensors are ``[C, H, W]``.
    """
    if volume.ndim == 5:
        volume = volume[0]
    if volume.ndim != 4:
        raise ValueError(
            f"Expected volume with shape [B,C,X,Y,Z] or [C,X,Y,Z], got {tuple(volume.shape)}"
        )
    if axis not in {"sagittal", "coronal", "axial"}:
        raise ValueError(f"Unknown sampling axis {axis!r}")

    spatial_axis = {"sagittal": 1, "coronal": 2, "axial": 3}[axis]
    axis_size = volume.shape[spatial_axis]
    results = []
    for s in slices:
        if not 0 <= s <= 1:
            raise ValueError(f"Slice fraction must be in [0, 1], got {s}")
        idx = min(int(axis_size * s), axis_size - 1)
        if axis == "sagittal":
            results.append(volume[:, idx, :, :])
        elif axis == "coronal":
            results.append(volume[:, :, idx, :])
        else:
            results.append(volume[:, :, :, idx])
    return results


@torch.inference_mode()
def decode_sliding(
    decoder: nn.Module,
    input: torch.Tensor,
    codebook: Optional[Union[nn.Module, str]] = None,
    heart_size: Tuple[int, int, int] = (80, 80, 80),
    window_size: Tuple[int, int, int] = (160, 160, 160),
    quantize: bool = False,
    denormalize: bool = False,
    latent_transform: Optional[LatentTransform] = None,
    gaussian_sigma: float = 0.5,
) -> torch.Tensor:
    """
    Ported Sliding Window Decoding logic.

    Args:
        decoder: The VAE decoder model (e.g., CTDecoder)
        input: Full-scan latent tensor of shape [B, C, D_lat, H_lat, W_lat]
        codebook: Optional codebook for quantization and de-normalization.
                  Can be a SemanticCodebook instance or a path to a checkpoint.
                  If a path is provided, it will load the codebook from that checkpoint.
        heart_size: The size of the 'perfectly healed' core in image space.
                    Determines the stride. Default (80, 80, 80).
        window_size: The size of the decoding window in image space.
                     Default (160, 160, 160).
        quantize: Whether to quantize the latent using the codebook before decoding.
        denormalize: Explicit legacy min/max compatibility switch.
        gaussian_sigma: Standard deviation of the Gaussian blending window.
        latent_transform: Versioned transform to invert before quantization and
            decoding. Mutually exclusive with ``denormalize``.

    Returns:
        Reconstructed full-volume scan as a Tensor.
    """
    if denormalize and latent_transform is not None:
        raise ValueError("denormalize and latent_transform are mutually exclusive")
    if gaussian_sigma <= 0:
        raise ValueError(f"gaussian_sigma must be positive, got {gaussian_sigma}")
    device = input.device
    b, c_lat, d_lat, h_lat, w_lat = input.shape

    # Determine upsample ratio from decoder config
    # Default to 8 if not found
    upscale = 8
    embedding_dim = 8
    n_codes = 16384
    if hasattr(decoder, "config"):
        upscale = 8 if decoder.config.scale == "8X" else 4
        embedding_dim = getattr(decoder.config, "embedding_dim", 8)
        n_codes = getattr(decoder.config, "n_codes", 16384)

    # Handle codebook loading if a path is provided
    if isinstance(codebook, str):
        logger.info(f"Loading codebook from checkpoint: {codebook}")
        codebook = SemanticCodebook.load_from_checkpoint(
            codebook,
            n_codes=n_codes,
            embedding_dim=embedding_dim,
            assignment_only_l2=getattr(
                decoder.config, "assignment_only_l2", False
            ),
            spherical_vq=getattr(decoder.config, "spherical_vq", False),
        ).to(device)
        codebook.eval()

    if latent_transform is not None:
        logger.info(f"Inverting latent transform {latent_transform.transform_type}")
        input = latent_transform.inverse(input)
    elif denormalize and codebook is not None:
        logger.info("Applying de-normalization using codebook stats")
        cb_obj = codebook.codebook if hasattr(codebook, "codebook") else codebook
        embeddings = cb_obj.embeddings
        e_min = embeddings.min()
        e_max = embeddings.max()
        input = (((input + 1.0) / 2.0) * (e_max - e_min)) + e_min

    # Target image size
    d_img, h_image, w_img = d_lat * upscale, h_lat * upscale, w_lat * upscale

    # Latent window and stride
    # window_size 160 -> 20 in latent
    # heart_size 80 -> 10 in latent (stride)
    win_lat = [s // upscale for s in window_size]
    stride_lat = [s // upscale for s in heart_size]

    # Prepare buffers on CPU to save GPU memory for large scans
    # We will accumulate everything here
    output_volume = torch.zeros(
        (b, decoder.config.image_channels, d_img, h_image, w_img), device="cpu"
    )
    weight_volume = torch.zeros((b, 1, d_img, h_image, w_img), device="cpu")

    # Prepare Gaussian weight mask for the window
    # Values near 1.0 in center, fading to 0.0 at edges
    weight_mask = get_gaussian_window(window_size, sigma=gaussian_sigma).to("cpu")

    logger.info(
        f"Starting sliding window decode: Latent {input.shape} -> Image {(d_img, h_image, w_img)}, Quantize={quantize}, Denormalize={denormalize}"
    )

    # Slide through the latent space
    for z_start in get_sliding_window_starts(d_lat, win_lat[0], stride_lat[0]):
        z_end = z_start + win_lat[0]

        for y_start in get_sliding_window_starts(h_lat, win_lat[1], stride_lat[1]):
            y_end = y_start + win_lat[1]

            for x_start in get_sliding_window_starts(w_lat, win_lat[2], stride_lat[2]):
                x_end = x_start + win_lat[2]

                # Extract latent window
                latent_window = input[:, :, z_start:z_end, y_start:y_end, x_start:x_end]

                if quantize and codebook is not None:
                    latent_window = codebook(latent_window)["embeddings"]

                # Decode on GPU
                decoded_window = decoder(latent_window).detach().cpu()

                # Calculate image space coordinates
                zi, yi, xi = z_start * upscale, y_start * upscale, x_start * upscale
                zi_e, yi_e, xi_e = (
                    zi + window_size[0],
                    yi + window_size[1],
                    xi + window_size[2],
                )

                # Accumulate
                # Note: We use the full weight mask. If the window was smaller than window_size
                # (due to tiny latents), we'd need to crop the mask, but win_lat is fixed here.
                output_volume[:, :, zi:zi_e, yi:yi_e, xi:xi_e] += (
                    decoded_window * weight_mask
                )
                weight_volume[:, :, zi:zi_e, yi:yi_e, xi:xi_e] += weight_mask

                # Free GPU memory
                del decoded_window
                torch.cuda.empty_cache()

    # Final blending
    # Avoid division by zero
    weight_volume[weight_volume == 0] = 1.0
    reconstructed = output_volume / weight_volume

    return reconstructed.to(device)
