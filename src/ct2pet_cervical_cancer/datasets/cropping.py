"""Align CT/PET/mask grids, then crop their RAS axial extent."""
import argparse
import json
from pathlib import Path
from typing import Any
import nibabel as nib
import numpy as np
import torch
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Resized, ResampleToMatchd

TARGET_SIZE = (320, 320, 320)

def build_alignment_transform() -> Compose:
    """Build the same spatial alignment used by the PVAE test script."""
    return Compose(
        [
            LoadImaged(keys=("ct", "pet", "mask")),
            EnsureChannelFirstd(keys=("ct", "pet", "mask")),
            Orientationd(keys=("ct", "pet", "mask"), axcodes="RAS"),
            Resized(
                keys="ct",
                spatial_size=TARGET_SIZE,
                mode="trilinear",
            ),
            ResampleToMatchd(
                keys="pet",
                key_dst="ct",
                mode="bilinear",
                padding_mode="zeros",
            ),
            ResampleToMatchd(
                keys="mask",
                key_dst="ct",
                mode="nearest",
                padding_mode="zeros",
            ),
        ]
    )


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def find_mask_bounds(mask: Any) -> tuple[int, int, int]:
    """Return inclusive/exclusive bounds on the RAS axial (last) axis."""
    mask_array = _as_numpy(mask)
    if mask_array.ndim == 4:
        mask_array = mask_array[0]
    if mask_array.ndim != 3:
        raise ValueError(f"Expected a 3D mask after channel removal, got {mask_array.shape}")

    # After Orientationd(..., axcodes='RAS'), the axial/superior-inferior
    # dimension is the last array axis. The first two axes are collapsed here.
    nonempty = np.any(np.nan_to_num(mask_array) != 0, axis=(0, 1))
    indices = np.flatnonzero(nonempty)
    if indices.size == 0:
        raise ValueError("total_mask.nii.gz contains no non-zero axial layer")
    return int(indices[0]), int(indices[-1]) + 1, int(indices.size)


def expand_bounds(
    start: int,
    end: int,
    total_depth: int,
    min_depth: int,
    depth_multiple: int,
    margin: int,
) -> tuple[int, int]:
    if min_depth < 1:
        raise ValueError(f"min_depth must be positive, got {min_depth}")
    if depth_multiple < 1:
        raise ValueError(f"depth_multiple must be positive, got {depth_multiple}")
    if margin < 0:
        raise ValueError(f"margin cannot be negative, got {margin}")

    start = max(0, start - margin)
    end = min(total_depth, end + margin)
    required_depth = max(min_depth, end - start)
    required_depth = int(np.ceil(required_depth / depth_multiple) * depth_multiple)
    required_depth = min(required_depth, total_depth)

    if end - start < required_depth:
        extra = required_depth - (end - start)
        start = max(0, start - extra // 2)
        end = min(total_depth, end + extra - extra // 2)

    # If clipping at one boundary prevented the first symmetric expansion,
    # shift the opposite boundary so the requested depth is reached whenever
    # the aligned volume is large enough.
    current = end - start
    if current < required_depth:
        if start == 0:
            end = min(total_depth, end + required_depth - current)
        elif end == total_depth:
            start = max(0, start - (required_depth - current))

    return start, end


def crop_affine(affine: np.ndarray, axial_start: int) -> np.ndarray:
    updated = np.asarray(affine, dtype=np.float64).copy()
    updated[:3, 3] += updated[:3, 2] * axial_start
    return updated


def save_nifti(value: Any, affine: np.ndarray, path: Path) -> None:
    array = _as_numpy(value)
    if array.ndim == 4:
        array = array[0]
    nib.save(nib.Nifti1Image(array.astype(np.float32, copy=False), affine), str(path))


def process_case(
    case_dir: Path,
    seg_dir: Path,
    output_root: Path,
    align: Compose,
    min_depth: int,
    depth_multiple: int,
    margin: int,
    overwrite: bool,
) -> dict[str, Any]:
    ct_path = case_dir / "ct.nii.gz"
    pet_path = case_dir / "pet.nii.gz"
    mask_path = seg_dir / "total_mask.nii.gz"
    missing = [str(path) for path in (ct_path, pet_path, mask_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))

    output_case = output_root / case_dir.name
    if output_case.exists() and not overwrite:
        return {"case": case_dir.name, "status": "skipped", "reason": "output exists"}
    output_case.mkdir(parents=True, exist_ok=True)

    aligned = align({"ct": str(ct_path), "pet": str(pet_path), "mask": str(mask_path)})
    ct, pet, mask = aligned["ct"], aligned["pet"], aligned["mask"]
    aligned_shape = tuple(int(size) for size in ct.shape[1:])
    if aligned_shape != TARGET_SIZE:
        raise ValueError(f"Aligned CT shape is {aligned_shape}, expected {TARGET_SIZE}")

    raw_start, raw_end, nonempty_layers = find_mask_bounds(mask)
    start, end = expand_bounds(
        raw_start,
        raw_end,
        aligned_shape[2],
        min_depth,
        depth_multiple,
        margin,
    )
    cropped_affine = crop_affine(_as_numpy(ct.affine), start)
    ct_crop = ct[..., start:end]
    pet_crop = pet[..., start:end]
    mask_crop = mask[..., start:end]

    save_nifti(ct_crop, cropped_affine, output_case / "ct.nii.gz")
    save_nifti(pet_crop, cropped_affine, output_case / "pet.nii.gz")
    save_nifti(mask_crop, cropped_affine, output_case / "total_mask.nii.gz")

    metadata = {
        "case": case_dir.name,
        "source_ct": str(ct_path),
        "source_pet": str(pet_path),
        "source_mask": str(mask_path),
        "target_alignment_shape": list(TARGET_SIZE),
        "axis_convention": "RAS; axial crop is the last array axis",
        "mask_nonempty_bounds_aligned": [raw_start, raw_end],
        "crop_bounds_aligned": [start, end],
        "nonempty_axial_layers": nonempty_layers,
        "output_shape": [int(size) for size in ct_crop.shape[1:]],
        "min_depth": min_depth,
        "depth_multiple": depth_multiple,
        "margin": margin,
    }
    with (output_case / "crop_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return {
        "case": case_dir.name,
        "status": "processed",
        "mask_bounds": [raw_start, raw_end],
        "crop_bounds": [start, end],
        "output_shape": metadata["output_shape"],
    }


def process_case_safe(
    case_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Process one case with an independent transform instance.

    A separate Compose instance avoids sharing transform state between worker
    threads. PyTorch intra-op threads are limited in ``main`` when workers are
    enabled, preventing CPU oversubscription.
    """
    if not case_dir.is_dir():
        return {"case": case_dir.name, "status": "failed", "reason": "case not found"}
    try:
        return process_case(
            case_dir=case_dir,
            seg_dir=args.seg_root / case_dir.name,
            output_root=args.output_root,
            align=build_alignment_transform(),
            min_depth=args.min_depth,
            depth_multiple=args.depth_multiple,
            margin=args.margin,
            overwrite=args.overwrite,
        )
    except Exception as exc:  # keep independent cases running
        if args.fail_fast:
            raise
        return {"case": case_dir.name, "status": "failed", "reason": str(exc)}
