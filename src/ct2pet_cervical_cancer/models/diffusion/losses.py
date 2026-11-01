"""Image-space supervision through a frozen, differentiable PET decoder."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from ...datasets.latent_transform import LatentTransform
from ...datasets.paired_latents import stable_seed
from ...datasets.lesion_sampling import LesionPatchSampler, LesionROIPatchSampler
from ..vae.loading import load_config
from ..vae.pet import PETDecoder, PETEncoderDecoderConfig


IMAGE_L1_DEFAULTS = {"image_l1_weight": 0.0, "image_l1_target": "vae_reconstruction",
                     "image_l1_timestep_weighting": "linear", "image_l1_patch_size": 80,
                     "lesion_mask_root": None, "lesion_sampling_probability": 0.0}
LESION_L1_DEFAULTS = {"lesion_l1_weight": 0.0, "lesion_l1_patch_size": 80,
                      "lesion_l1_timestep_weighting": "linear"}


class ImageL1Loss(nn.Module):
    """One corresponding image patch per case, in decoder intensity coordinates.

    The reconstruction target is D(z0), not the original PET. Paired NIfTI
    supervision instead clips PET to [0,5] and maps it to [-1,1]. Predictions
    are never quantized or clipped. Decoder parameters stay frozen while
    gradients flow through the prediction branch to the diffusion model.
    """
    def __init__(self, decoder: nn.Module, transform: LatentTransform, patch_size: int = 80,
                 target: str = "vae_reconstruction", timestep_weighting: str = "linear",
                 reference_root: Path | None = None, lesion_mask_root: Path | None = None,
                 lesion_sampling_probability: float = 0.0):
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
        if not np.isfinite(lesion_sampling_probability) or not 0 <= lesion_sampling_probability <= 1:
            raise ValueError("lesion_sampling_probability must be between zero and one")
        if lesion_sampling_probability > 0 and lesion_mask_root is None:
            raise ValueError("Lesion sampling requires lesion_mask_root")
        self.patch_sampler = (LesionPatchSampler(lesion_mask_root, lesion_sampling_probability, self.upscale)
                              if lesion_sampling_probability > 0 else None)

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

    def decode_prediction(self, raw_prediction: torch.Tensor) -> torch.Tensor:
        return (checkpoint(self.decoder, raw_prediction, use_reentrant=False)
                if torch.is_grad_enabled() and raw_prediction.requires_grad else self.decoder(raw_prediction))

    def forward(self, prediction: torch.Tensor, clean: torch.Tensor, timesteps: torch.Tensor,
                num_timesteps: int, cases: list[str], seed: int | None = None) -> dict:
        if prediction.shape != clean.shape or prediction.ndim != 5 or len(cases) != len(clean):
            raise ValueError("Image L1 requires matching batched PET latents and case IDs")
        sizes = tuple(min(size, self.patch_size) for size in clean.shape[2:])
        errors = []
        selected_count = empty_count = 0
        # Validation/test keep their original fixed random patches and never read masks.
        mixed_sampling = self.patch_sampler is not None and seed is None
        # Keep the affine inverse and decoder in float32, also under AMP.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            for index, case in enumerate(cases):
                generator = (torch.Generator(device=prediction.device).manual_seed(
                    stable_seed(seed, f"image_l1:{case}")) if seed is not None else None)
                if mixed_sampling:
                    starts, selected, empty = self.patch_sampler.sample(
                        case, tuple(clean.shape[2:]), sizes, prediction.device, generator)
                    selected_count += selected
                    empty_count += empty
                else:
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
                decoded = self.decode_prediction(raw_prediction)
                if decoded.shape != target.shape:
                    raise ValueError("Decoded prediction and image L1 target shapes differ")
                errors.append((decoded.float() - target.float()).abs().mean())
            per_case = torch.stack(errors)
            weights = ((num_timesteps - timesteps.float()) / num_timesteps
                       if self.timestep_weighting == "linear" else torch.ones_like(timesteps, dtype=torch.float32))
            result = {"image_l1": per_case.mean(), "image_l1_weighted": (weights * per_case).mean()}
            if mixed_sampling:
                result.update(image_l1_lesion_sampled_fraction=per_case.new_tensor(selected_count / len(cases)),
                              image_l1_empty_mask_fraction=per_case.new_tensor(empty_count / len(cases)))
            return result


class LesionL1Loss(ImageL1Loss):
    """Independent lesion patch L1 against actual PET, in decoder coordinates.

    Return sums over positive cases, not means over the whole batch. The
    trainer uses the global positive-case count for DDP gradient normalization.
    No collectives occur here, so evaluation may have uneven or empty shards.
    """
    def __init__(self, decoder: nn.Module, transform: LatentTransform, reference_root: Path,
                 lesion_mask_root: Path, patch_size: int = 80, timestep_weighting: str = "linear"):
        if reference_root is None or lesion_mask_root is None:
            raise ValueError("Lesion L1 requires reference_root and lesion_mask_root")
        super().__init__(decoder, transform, patch_size=patch_size, target="paired_nifti",
                         timestep_weighting=timestep_weighting, reference_root=reference_root)
        self.roi_sampler = LesionROIPatchSampler(lesion_mask_root, self.upscale)

    def forward(self, prediction: torch.Tensor, clean: torch.Tensor, timesteps: torch.Tensor,
                num_timesteps: int, cases: list[str], seed: int | None = None) -> dict:
        if prediction.shape != clean.shape or prediction.ndim != 5 or len(cases) != len(clean):
            raise ValueError("Lesion L1 requires matching batched PET latents and case IDs")
        sizes = tuple(min(size, self.patch_size) for size in clean.shape[2:])
        errors, weighted_errors = [], []
        voxel_count = 0
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            for index, case in enumerate(cases):
                generator = (torch.Generator(device=prediction.device).manual_seed(
                    stable_seed(seed, f"lesion_l1:{case}")) if seed is not None else None)
                selected = self.roi_sampler.sample(case, tuple(clean.shape[2:]), sizes,
                                                   prediction.device, generator)
                if selected is None:
                    continue
                starts, mask = selected
                slices = (slice(index, index + 1), slice(None)) + tuple(
                    slice(start, start + size) for start, size in zip(starts, sizes, strict=True))
                with torch.no_grad():
                    target = self.reference_patch(case, clean.shape[2:], starts, sizes, prediction.device)
                    roi = torch.from_numpy(mask).to(prediction.device, dtype=torch.bool)[None, None]
                decoded = self.decode_prediction(self.transform.inverse(prediction[slices].float()))
                if decoded.shape != target.shape or decoded.shape != roi.shape:
                    raise ValueError("Decoded lesion prediction, PET target and mask shapes differ")
                # Boolean indexing makes background voxels contribute exactly zero.
                error = (decoded.float() - target.float()).abs()[roi].mean()
                weight = ((num_timesteps - timesteps[index].float()) / num_timesteps
                          if self.timestep_weighting == "linear" else 1.0)
                errors.append(error)
                weighted_errors.append(error * weight)
                voxel_count += int(mask.sum())
            # An empty local rank must still permit a backward pass. Its zero term
            # is independent of decoder loading/calls and of the positive ranks.
            zero = prediction.float().sum() * 0
            return {"lesion_l1_sum": torch.stack(errors).sum() if errors else zero,
                    "lesion_l1_weighted_sum": torch.stack(weighted_errors).sum() if errors else zero,
                    "lesion_l1_case_count": len(errors), "lesion_l1_voxel_count": voxel_count}
