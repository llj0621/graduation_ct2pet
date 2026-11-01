"""Mask-guided image-loss patches; diffusion still receives the full latent."""
from pathlib import Path
import math
import nibabel as nib
import numpy as np
import torch


class LesionPatchSampler:
    def __init__(self, mask_root: Path, probability: float, upscale: int):
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("lesion_sampling_probability must be between zero and one")
        self.mask_root = Path(mask_root)
        self.probability, self.upscale = probability, upscale
        self._coordinates: dict[str, tuple[tuple, np.ndarray]] = {}

    @staticmethod
    def load_mask(path: Path, expected_shape: tuple,
                  reference_path: Path | None = None) -> np.ndarray:
        image = nib.load(str(path))
        if image.shape != expected_shape:
            raise ValueError(f"Lesion mask shape differs for {path.name}: {image.shape}, expected {expected_shape}")
        mask = np.asanyarray(image.dataobj)
        if not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all():
            raise ValueError(f"Lesion mask must contain finite binary 0/1 labels: {path}")
        if reference_path is not None:
            reference = nib.load(str(reference_path))
            if (reference.shape != image.shape
                    or not np.allclose(reference.affine, image.affine, rtol=1e-6, atol=1e-4)):
                raise ValueError(f"Lesion mask reference grid differs: {path}")
        return mask > 0

    def coordinates(self, case: str, spatial_shape: tuple) -> np.ndarray:
        if case not in self._coordinates:
            expected = tuple(length * self.upscale for length in spatial_shape)
            mask = self.load_mask(self.mask_root / f"{case}.nii.gz", expected)
            # Any positive image voxel keeps its latent cell eligible, including tiny lesions.
            x, y, z = spatial_shape
            coarse = mask.reshape(x, self.upscale, y, self.upscale, z, self.upscale).any(axis=(1, 3, 5))
            self._coordinates[case] = (spatial_shape, np.argwhere(coarse))
        shape, coordinates = self._coordinates[case]
        if shape != spatial_shape:
            raise ValueError(f"Latent shape changed for mask sampling: {case}")
        return coordinates

    def sample(self, case: str, spatial_shape: tuple, sizes: tuple,
               device: torch.device, generator: torch.Generator | None = None) -> tuple[tuple, bool, bool]:
        """Return aligned latent starts, lesion-branch flag and empty-mask flag.

        Pick an occupied latent cell, then randomize a legal crop containing it.
        This preserves context and covers lesions at volume edges without resizing.
        """
        coordinates = self.coordinates(case, spatial_shape)
        selected = bool(len(coordinates)) and bool(
            torch.rand((), device=device, generator=generator) < self.probability)
        if selected:
            index = int(torch.randint(len(coordinates), (), device=device, generator=generator))
            anchor = coordinates[index]
            bounds = [(max(0, int(point) - size + 1), min(int(point), length - size))
                      for point, length, size in zip(anchor, spatial_shape, sizes, strict=True)]
        else:
            bounds = [(0, length - size) for length, size in zip(spatial_shape, sizes, strict=True)]
        starts = tuple(int(torch.randint(low, high + 1, (), device=device, generator=generator))
                       for low, high in bounds)
        return starts, selected, not bool(len(coordinates))


class LesionROIPatchSampler:
    """Choose a component uniformly, then a scale-aligned patch containing it.

    Cache sparse image coordinates rather than full 320x320x80 masks. The
    returned mask includes every annotated lesion voxel inside the selected
    patch, including components other than the one used to choose its location.
    """
    def __init__(self, mask_root: Path, upscale: int):
        self.mask_root, self.upscale = Path(mask_root), upscale
        self._components: dict[str, tuple[tuple, tuple[np.ndarray, ...]]] = {}

    def components(self, case: str, spatial_shape: tuple) -> tuple[np.ndarray, ...]:
        if case not in self._components:
            from scipy import ndimage
            expected = tuple(length * self.upscale for length in spatial_shape)
            mask = LesionPatchSampler.load_mask(self.mask_root / f"{case}.nii.gz", expected)
            labels, count = ndimage.label(mask, ndimage.generate_binary_structure(3, 3))
            coordinates = np.argwhere(mask).astype(np.int32)
            ids = labels[tuple(coordinates.T)]
            components = tuple(coordinates[ids == index] for index in range(1, count + 1))
            self._components[case] = (spatial_shape, components)
        shape, components = self._components[case]
        if shape != spatial_shape:
            raise ValueError(f"Latent shape changed for ROI sampling: {case}")
        return components

    def sample(self, case: str, spatial_shape: tuple, sizes: tuple,
               device: torch.device, generator: torch.Generator | None = None) -> tuple | None:
        components = self.components(case, spatial_shape)
        if not components:
            return None
        component = components[int(torch.randint(len(components), (), device=device, generator=generator))]
        point = component[int(torch.randint(len(component), (), device=device, generator=generator))]
        anchor = point // self.upscale
        bounds = [(max(0, int(p) - size + 1), min(int(p), length - size))
                  for p, length, size in zip(anchor, spatial_shape, sizes, strict=True)]
        starts = tuple(int(torch.randint(low, high + 1, (), device=device, generator=generator))
                       for low, high in bounds)
        origin = np.asarray(starts) * self.upscale
        image_sizes = np.asarray(sizes) * self.upscale
        mask = np.zeros(tuple(image_sizes), dtype=np.uint8)
        for coordinates in components:
            local = coordinates - origin
            inside = np.all((local >= 0) & (local < image_sizes), axis=1)
            mask[tuple(local[inside].T)] = 1
        return starts, mask
