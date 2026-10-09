import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from loguru import logger
from ..datasets.latent_transform import LatentTransform

__all__ = ("get_gaussian_window", "get_sliding_window_starts", "encode_sliding")


def get_gaussian_window(
    window_size: Tuple[int, int, int], sigma: float = 0.5
) -> torch.Tensor:
    """
    Generates a 3D Gaussian window for blending.
    """
    z, y, x = window_size
    z_lin = torch.linspace(-1, 1, z)
    y_lin = torch.linspace(-1, 1, y)
    x_lin = torch.linspace(-1, 1, x)

    z_g = torch.exp(-0.5 * (z_lin / sigma) ** 2)
    y_g = torch.exp(-0.5 * (y_lin / sigma) ** 2)
    x_g = torch.exp(-0.5 * (x_lin / sigma) ** 2)

    # Create 3D grid
    grid = z_g.view(-1, 1, 1) * y_g.view(1, -1, 1) * x_g.view(1, 1, -1)
    return grid


def get_sliding_window_starts(length: int, window: int, stride: int) -> list[int]:
    """Return unique sliding-window starts, including exactly one final window."""
    if length < 1 or window < 1 or stride < 1:
        raise ValueError("length, window, and stride must be positive")
    if window >= length:
        return [0]
    final_start = length - window
    starts = list(range(0, final_start + 1, stride))
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


@torch.inference_mode()
def encode_sliding(
    image: torch.Tensor,
    encoder: nn.Module,
    codebook: Optional[nn.Module] = None,
    resize: Tuple[int, int, int] = (320, 320, 320),
    roi: Tuple[int, int, int] = (80, 80, 80),
    heart_size: Tuple[int, int, int] = (40, 40, 40),
    quantize: bool = False,
    normalize: bool = False,
    latent_transform: Optional[LatentTransform] = None,
    gaussian_sigma: float = 0.5,
) -> torch.Tensor:
    """
    Encodes a full 3D scan using an overlapping sliding window to reduce latent-space seams.

    Args:
        image: Input 3D scan tensor [B, C, D, H, W]
        encoder: The encoder model (e.g., CTEncoder)
        codebook: Optional codebook for quantization and normalization.
        resize: Target size to resize the full scan before processing.
        roi: Size of the windows to extract and encode.
        heart_size: The stride for the sliding window in image space.
        quantize: Whether to quantize the latent using the codebook.
        normalize: Legacy codebook min/max normalization switch.
        gaussian_sigma: Standard deviation of the Gaussian blending window.
        latent_transform: Explicit versioned PVAE-to-diffusion transform. This is
            mutually exclusive with ``normalize``.

    Returns:
        Full-scan latent tensor.
    """
    if normalize and latent_transform is not None:
        raise ValueError("normalize and latent_transform are mutually exclusive")
    if gaussian_sigma <= 0:
        raise ValueError(f"gaussian_sigma must be positive, got {gaussian_sigma}")
    device = image.device
    b, c, d, h, w = image.shape

    # 1. Resize to target resolution
    if (d, h, w) != resize:
        logger.info(f"Resizing input scan from {(d, h, w)} to {resize}")
        image = F.interpolate(image, size=resize, mode="trilinear", align_corners=False)

    d_img, h_img, w_img = resize

    # Determine downsample ratio
    upscale = 8
    if hasattr(encoder, "config"):
        upscale = 8 if encoder.config.scale == "8X" else 4

    # Calculate target latent dimensions
    d_lat, h_lat, w_lat = d_img // upscale, h_img // upscale, w_img // upscale
    embedding_dim = encoder.config.embedding_dim if hasattr(encoder, "config") else 8

    # 2. Prepare Accumulation Buffers on CPU
    full_latent = torch.zeros((b, embedding_dim, d_lat, h_lat, w_lat), device="cpu")
    weight_buffer = torch.zeros((b, 1, d_lat, h_lat, w_lat), device="cpu")

    # 3. Prepare Gaussian Window
    # The weight mask is applied in the LATENT space because we are reassembling latents.
    latent_roi = [s // upscale for s in roi]
    weight_mask = get_gaussian_window(
        tuple(latent_roi), sigma=gaussian_sigma
    ).to("cpu")

    logger.info(
        f"Encoding with overlapping windows: ROI {roi}, Stride {heart_size}, Quantize={quantize}, Normalize={normalize}..."
    )

    # 4. Slide through the Image space
    for z_start in get_sliding_window_starts(d_img, roi[0], heart_size[0]):
        z_end = z_start + roi[0]

        for y_start in get_sliding_window_starts(h_img, roi[1], heart_size[1]):
            y_end = y_start + roi[1]

            for x_start in get_sliding_window_starts(w_img, roi[2], heart_size[2]):
                x_end = x_start + roi[2]

                # Extract image patch
                sub_roi = image[:, :, z_start:z_end, y_start:y_end, x_start:x_end]

                # Encode on GPU
                enc_out = encoder(sub_roi)
                h_feat = enc_out["embeddings"] if isinstance(enc_out, dict) else enc_out

                if quantize and codebook is not None:
                    h_feat = codebook(h_feat)["embeddings"]

                h_feat = h_feat.detach().cpu()

                # Map image coordinates to latent coordinates
                zs, ys, xs = z_start // upscale, y_start // upscale, x_start // upscale
                ze, ye, xe = (
                    zs + latent_roi[0],
                    ys + latent_roi[1],
                    xs + latent_roi[2],
                )

                # Weighted accumulation in latent space
                full_latent[:, :, zs:ze, ys:ye, xs:xe] += h_feat * weight_mask
                weight_buffer[:, :, zs:ze, ys:ye, xs:xe] += weight_mask

                del h_feat
                torch.cuda.empty_cache()

    # 5. Final Normalization
    weight_buffer[weight_buffer == 0] = 1.0
    full_latent = full_latent / weight_buffer

    if latent_transform is not None:
        logger.info(f"Applying latent transform {latent_transform.transform_type}")
        full_latent = latent_transform.forward(full_latent.to(device))
    elif normalize and codebook is not None:
        logger.info("Applying [-1, 1] normalization using codebook stats")
        # Access the underlying codebook object if it's wrapped
        cb_obj = codebook.codebook if hasattr(codebook, "codebook") else codebook
        embeddings = cb_obj.embeddings
        e_min = embeddings.min()
        e_max = embeddings.max()

        # Move the latent to GPU for the final scaling operation
        full_latent = full_latent.to(device)
        full_latent = ((full_latent - e_min) / (e_max - e_min)) * 2.0 - 1.0

    logger.info(f"Encoded full latent with shape {full_latent.shape}")
    return full_latent.to(device)
