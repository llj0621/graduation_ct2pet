"""Image-space supervision through a frozen, differentiable PET decoder."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from ...datasets.latent_transform import LatentTransform
from ...datasets.paired_latents import stable_seed
from ..vae.loading import load_config
from ..vae.pet import PETDecoder, PETEncoderDecoderConfig


IMAGE_L1_DEFAULTS = {"image_l1_weight": 0.0, "image_l1_target": "vae_reconstruction",
                     "image_l1_timestep_weighting": "linear", "image_l1_patch_size": 80}


class ImageL1Loss(nn.Module):
    """One corresponding image patch per case, in decoder intensity coordinates.

    The reconstruction target is D(z0), not the original PET. Paired NIfTI
    supervision instead clips PET to [0,5] and maps it to [-1,1]. Predictions
    are never quantized or clipped. Decoder parameters stay frozen while
    gradients flow through the prediction branch to the diffusion model.
    """
    def __init__(self, decoder: nn.Module, transform: LatentTransform, patch_size: int = 80,
                 target: str = "vae_reconstruction", timestep_weighting: str = "linear",
                 reference_root: Path | None = None):
        super().__init__()
        self.decoder = decoder.eval().requires_grad_(False)
        self.transform = transform
        self.upscale = 8 if decoder.config.scale == "8X" else 4
        if patch_size < self.upscale or patch_size % self.upscale:
            raise ValueError("image_l1_patch_size must be a positive multiple of the PET decoder scale")
        if target not in {"vae_reconstruction", "paired_nifti"}:
            raise ValueError("Unknown image_l1_target")
        if timestep_weighting not in {"none", "linear"}:
            raise ValueError("Unknown image_l1_timestep_weighting")
        if target == "paired_nifti" and reference_root is None:
            raise ValueError("paired_nifti image L1 requires reference_root")
        self.patch_size = patch_size // self.upscale
        self.target, self.timestep_weighting = target, timestep_weighting
        self.reference_root = reference_root

    @classmethod
    def from_checkpoint(cls, path: Path, transform: LatentTransform, device: torch.device, **kwargs):
        config = load_config(path, PETEncoderDecoderConfig)
        decoder = PETDecoder.load_from_checkpoint(str(path), config).to(device)
        return cls(decoder, transform, **kwargs)

    def reference_patch(self, case: str, spatial_shape: tuple, starts: tuple, sizes: tuple,
                        device: torch.device) -> torch.Tensor:
        import nibabel as nib
        path = self.reference_root / case / "pet.nii.gz"
        image = nib.load(str(path))
        expected = tuple(size * self.upscale for size in spatial_shape)
        if image.shape != expected:
            raise ValueError(f"PET reference grid differs for {case}: {image.shape}, expected {expected}")
        slices = tuple(slice(start * self.upscale, (start + size) * self.upscale)
                       for start, size in zip(starts, sizes, strict=True))
        patch = np.array(image.dataobj[slices], dtype=np.float32, copy=True)
        if not np.isfinite(patch).all():
            raise ValueError(f"Non-finite PET reference in {path}")
        return torch.from_numpy(patch).to(device)[None, None].clamp(0, 5) / 2.5 - 1

    def forward(self, prediction: torch.Tensor, clean: torch.Tensor, timesteps: torch.Tensor,
                num_timesteps: int, cases: list[str], seed: int | None = None) -> dict:
        if prediction.shape != clean.shape or prediction.ndim != 5 or len(cases) != len(clean):
            raise ValueError("Image L1 requires matching batched PET latents and case IDs")
        sizes = tuple(min(size, self.patch_size) for size in clean.shape[2:])
        errors = []
        # Keep the affine inverse and decoder in float32, also under AMP.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            for index, case in enumerate(cases):
                generator = (torch.Generator(device=prediction.device).manual_seed(
                    stable_seed(seed, f"image_l1:{case}")) if seed is not None else None)
                starts = tuple(int(torch.randint(length - size + 1, (), device=prediction.device,
                               generator=generator)) for length, size in zip(clean.shape[2:], sizes, strict=True))
                slices = (slice(index, index + 1), slice(None)) + tuple(
                    slice(start, start + size) for start, size in zip(starts, sizes, strict=True))
                raw_prediction = self.transform.inverse(prediction[slices].float())
                with torch.no_grad():
                    if self.target == "vae_reconstruction":
                        target = self.decoder(self.transform.inverse(clean[slices].float()))
                    else:
                        target = self.reference_patch(case, clean.shape[2:], starts, sizes, prediction.device)
                decoded = (checkpoint(self.decoder, raw_prediction, use_reentrant=False)
                           if torch.is_grad_enabled() and raw_prediction.requires_grad else self.decoder(raw_prediction))
                if decoded.shape != target.shape:
                    raise ValueError("Decoded prediction and image L1 target shapes differ")
                errors.append((decoded.float() - target.float()).abs().mean())
            per_case = torch.stack(errors)
            weights = ((num_timesteps - timesteps.float()) / num_timesteps
                       if self.timestep_weighting == "linear" else torch.ones_like(timesteps, dtype=torch.float32))
            return {"image_l1": per_case.mean(), "image_l1_weighted": (weights * per_case).mean()}
