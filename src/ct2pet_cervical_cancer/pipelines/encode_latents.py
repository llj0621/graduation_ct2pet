"""Encode cropped physical CT/PET NIfTI pairs into continuous PVAE latents."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import torch
from ..models.vae.loading import load_encoder
from ..datasets.nifti import load_scan, save_latent
from ..inference.encode import encode_sliding

from ..utils.cli import parse_config_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--ct-checkpoint", type=Path, default=None)
    parser.add_argument("--pet-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--encode-stride",
        type=int,
        default=40,
        choices=(40, 80),
        help="Image-space stride for the encoder sliding windows; default: 40.",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=0.3,
        help="Sigma for latent-space Gaussian blending; default: 0.3.",
    )
    parser.add_argument(
        "--case",
        dest="case_name",
        default=None,
        help="Encode only one case directory; default: encode all cases.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing ct.pt/pet.pt pair.",
    )
    return parse_config_args(parser, "encode_latents.json")


@torch.inference_mode()
def encode_scan(
    image: torch.Tensor,
    encoder: torch.nn.Module,
    encode_stride: int,
    gaussian_sigma: float,
) -> torch.Tensor:
    spatial_size = tuple(int(size) for size in image.shape[2:])
    if min(spatial_size) < 80:
        raise ValueError(
            f"All spatial dimensions must be at least 80 for the current encoder window, "
            f"got {spatial_size}"
        )
    latent = encode_sliding(
        image=image,
        encoder=encoder,
        codebook=None,
        resize=spatial_size,
        roi=(80, 80, 80),
        heart_size=(encode_stride,) * 3,
        quantize=False,
        normalize=False,
        gaussian_sigma=gaussian_sigma,
    )
    # Existing diffusion dataloaders expect [C,D,H,W], not [B,C,D,H,W].
    return latent[0].detach().cpu().float().contiguous()


def encode_case(
    case_dir: Path,
    ct_encoder: torch.nn.Module,
    pet_encoder: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    ct_checkpoint: Path,
    pet_checkpoint: Path,
) -> dict[str, Any]:
    ct_path = case_dir / "ct.nii.gz"
    pet_path = case_dir / "pet.nii.gz"
    if not ct_path.is_file() or not pet_path.is_file():
        missing = [str(path) for path in (ct_path, pet_path) if not path.is_file()]
        raise FileNotFoundError("Missing paired scan(s): " + ", ".join(missing))

    output_case = args.output_root / case_dir.name
    output_case.mkdir(parents=True, exist_ok=True)
    ct_latent_path = output_case / "ct.pt"
    pet_latent_path = output_case / "pet.pt"
    if (ct_latent_path.exists() or pet_latent_path.exists()) and not args.overwrite:
        return {"case": case_dir.name, "status": "skipped", "reason": "latent exists"}

    ct_image = load_scan(ct_path, "ct", device)
    pet_image = load_scan(pet_path, "pet", device)
    if ct_image.shape != pet_image.shape:
        raise ValueError(
            f"CT/PET shape mismatch for {case_dir.name}: "
            f"{tuple(ct_image.shape)} vs {tuple(pet_image.shape)}"
        )

    ct_latent = encode_scan(ct_image, ct_encoder, args.encode_stride, args.gaussian_sigma)
    pet_latent = encode_scan(pet_image, pet_encoder, args.encode_stride, args.gaussian_sigma)
    save_latent(ct_latent, ct_latent_path)
    save_latent(pet_latent, pet_latent_path)

    metadata = {
        "case": case_dir.name,
        "ct_scan": str(ct_path),
        "pet_scan": str(pet_path),
        "ct_latent": str(ct_latent_path),
        "pet_latent": str(pet_latent_path),
        "input_shape_bchw_d": [int(size) for size in ct_image.shape],
        "ct_latent_shape": [int(size) for size in ct_latent.shape],
        "pet_latent_shape": [int(size) for size in pet_latent.shape],
        "encoding": {
            "resize": [int(size) for size in ct_image.shape[2:]],
            "roi": [80, 80, 80],
            "stride": [args.encode_stride] * 3,
            "gaussian_sigma": args.gaussian_sigma,
            "quantize": False,
            "normalization": {
                "ct": "clip[-1000,1000] then scale to [-1,1]",
                "pet": "clip[0,5] then scale to [-1,1]",
            },
        },
        "ct_checkpoint": str(ct_checkpoint),
        "pet_checkpoint": str(pet_checkpoint),
        "format": "torch_tensor_pt_v1",
    }
    with (output_case / "latent_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return {
        "case": case_dir.name,
        "status": "encoded",
        "ct_latent": str(ct_latent_path),
        "pet_latent": str(pet_latent_path),
        "latent_shape": [int(size) for size in ct_latent.shape],
    }


def main() -> None:
    args = parse_args()
    if not args.input_root.is_dir():
        raise FileNotFoundError(args.input_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for path in (args.ct_checkpoint, args.pet_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.gaussian_sigma <= 0:
        raise ValueError(f"gaussian_sigma must be positive, got {args.gaussian_sigma}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu only for a slow run.")

    case_dirs = (
        [args.input_root / args.case_name]
        if args.case_name
        else sorted(path for path in args.input_root.iterdir() if path.is_dir())
    )
    device = torch.device(args.device)
    ct_config, ct_encoder = load_encoder(args.ct_checkpoint, "ct", device)
    pet_config, pet_encoder = load_encoder(args.pet_checkpoint, "pet", device)
    print(f"CT config: {ct_config}")
    print(f"PET config: {pet_config}")

    records: list[dict[str, Any]] = []
    for index, case_dir in enumerate(case_dirs, start=1):
        print(f"[{index}/{len(case_dirs)}] {case_dir.name}")
        try:
            record = encode_case(
                case_dir,
                ct_encoder,
                pet_encoder,
                device,
                args,
                args.ct_checkpoint,
                args.pet_checkpoint,
            )
        except Exception as exc:
            record = {"case": case_dir.name, "status": "failed", "reason": str(exc)}
            print(f"  FAILED: {exc}")
        else:
            print(f"  {record['status']}: {record.get('latent_shape', '')}")
        records.append(record)

    manifest = {
        "format": "leaper_paired_latent_manifest_v1",
        "input_root": str(args.input_root),
        "ct_checkpoint": str(args.ct_checkpoint),
        "pet_checkpoint": str(args.pet_checkpoint),
        "gaussian_sigma": args.gaussian_sigma,
        "encode_stride": args.encode_stride,
        "records": records,
    }
    manifest_path = args.output_root / "latent_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"Saved manifest to {manifest_path}")
