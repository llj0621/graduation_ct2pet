"""NIfTI loading, intensity normalization and reconstruction preprocessing."""
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Resized, ResampleToMatchd
from .scan_utils import (normalize_ct_scan, normalize_pet_scan, CapScanValued,
                         NormalizeScanValued, CapPETScanValued, NormalizePETScanValued)

def load_scan(path: Path, modality: str, device: torch.device) -> torch.Tensor:
    image = nib.load(str(path))
    array = np.asanyarray(image.dataobj).astype(np.float32, copy=False)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI at {path}, got {array.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0).unsqueeze(0)
    if modality == "ct":
        tensor = normalize_ct_scan(tensor, to_one_one=True)
    else:
        tensor = normalize_pet_scan(tensor, to_one_one=True)
    return tensor.to(device)


def save_latent(latent: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(latent, path)


def prepare_ct(path: Path):
    transformed = Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            CapScanValued(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            Resized(keys="image", spatial_size=(320, 320, 320), mode="trilinear"),
            NormalizeScanValued(keys="image", to_one_one=True),
        ]
    )({"image": str(path)})["image"]
    return transformed.unsqueeze(0), np.asarray(transformed.affine)


def prepare_pet(pet_path: Path, ct_path: Path):
    transformed = Compose(
        [
            LoadImaged(keys=("pet", "ct")),
            EnsureChannelFirstd(keys=("pet", "ct")),
            Orientationd(keys=("pet", "ct"), axcodes="RAS"),
            Resized(
                keys="ct",
                spatial_size=(320, 320, 320),
                mode="trilinear",
                align_corners=False,
            ),
            ResampleToMatchd(
                keys="pet", key_dst="ct", mode="bilinear", padding_mode="zeros"
            ),
            CapPETScanValued(keys="pet"),
            NormalizePETScanValued(keys="pet", to_one_one=True),
        ]
    )({"pet": str(pet_path), "ct": str(ct_path)})["pet"]
    return transformed.unsqueeze(0), np.asarray(transformed.affine)


def save_outputs(reconstruction, affine, modality: str, output_dir: Path) -> None:
    normalized = reconstruction[0, 0].detach().cpu().float().numpy()
    physical = (normalized + 1.0) * (1000.0 if modality == "ct" else 2.5)
    if modality == "ct":
        physical -= 1000.0
    nib.save(
        nib.Nifti1Image(physical.astype(np.float32), affine),
        output_dir / f"{modality}_reconstruction.nii.gz",
    )
