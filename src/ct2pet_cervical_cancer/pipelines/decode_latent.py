"""Shared single-volume CT/PET VAE latent reconstruction workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch

from ..inference.decode import decode_sliding
from ..models.vae.loading import load_ct_decoder, load_pet_decoder
from ..utils.cli import parse_config_args


def parse_args(modality: str, config_name: str) -> argparse.Namespace:
    if modality not in {"ct", "pet"}:
        raise ValueError(f"Unknown modality {modality!r}")
    parser = argparse.ArgumentParser(description=f"Reconstruct one {modality.upper()} NIfTI volume from a saved latent.")
    parser.add_argument("--input-root", type=Path, default=None,
                        help="Root containing <case>/<latent-name> folders.")
    parser.add_argument("--reference-root", type=Path, default=None,
                        help="Root containing the corresponding CT/PET NIfTI files.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--latent-name", default=f"{modality}.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gaussian-sigma", type=float, default=0.3)
    parser.add_argument("--quantize", action=argparse.BooleanOptionalAction, default=True,
                        help="Quantize latents with this VAE checkpoint's codebook.")
    parser.add_argument("--clip-output", action=argparse.BooleanOptionalAction, default=True,
                        help="Clip to the intensity range used to train the selected VAE.")
    parser.add_argument("--case", dest="case_name", default=None,
                        help="Reconstruct one named case; default processes all cases.")
    parser.add_argument("--overwrite", action="store_true")
    return parse_config_args(parser, config_name)


def load_latent(path: Path, device: torch.device) -> torch.Tensor:
    payload: Any = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        for key in ("latent", "ct_latent", "pet_latent", "sample", "tensor"):
            if key in payload:
                payload = payload[key]
                break
        else:
            raise ValueError(f"No supported tensor key in {path}")
    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}, got {type(payload).__name__}")
    if payload.ndim == 4:
        payload = payload.unsqueeze(0)
    if payload.ndim != 5 or payload.shape[0] < 1:
        raise ValueError(f"Expected [C,D,H,W] or [B,C,D,H,W] latent, got {tuple(payload.shape)}")
    if not torch.isfinite(payload).all():
        raise ValueError(f"Latent contains NaN or Inf: {path}")
    return payload.float().contiguous().to(device)


@torch.inference_mode()
def decode_case(case_dir: Path, reference_root: Path, output_root: Path,
                decoder, codebook, device: torch.device,
                args: argparse.Namespace, modality: str) -> dict[str, Any]:
    latent_path = case_dir / args.latent_name
    reference_path = reference_root / case_dir.name / f"{modality}.nii.gz"
    if not latent_path.is_file():
        raise FileNotFoundError(latent_path)
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)

    reference = nib.load(str(reference_path))
    latent = load_latent(latent_path, device)
    decoded = decode_sliding(
        decoder=decoder, input=latent, codebook=codebook,
        quantize=args.quantize, denormalize=False,
        window_size=(80, 80, 80), heart_size=(40, 40, 40),
        gaussian_sigma=args.gaussian_sigma,
    )
    if decoded.shape[1] != 1:
        raise ValueError(f"Expected one image channel, got {decoded.shape[1]}")
    if tuple(decoded.shape[2:]) != tuple(reference.shape):
        raise ValueError(
            f"Decoded shape {tuple(decoded.shape[2:])} does not match reference "
            f"shape {reference.shape} for {case_dir.name}"
        )

    scale = 1000.0 if modality == "ct" else 2.5
    output_range = (-1000.0, 1000.0) if modality == "ct" else (0.0, 5.0)
    output_case = output_root / case_dir.name
    output_case.mkdir(parents=True, exist_ok=True)
    names = ([f"{modality}_reconstruction.nii.gz"] if decoded.shape[0] == 1 else
             [f"{modality}_reconstruction_{i:03d}.nii.gz" for i in range(decoded.shape[0])])
    outputs = []
    for index, name in enumerate(names):
        output_path = output_case / name
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists (pass --overwrite to replace): {output_path}")
        normalized = decoded[index, 0].detach().cpu().float().numpy()
        if args.clip_output:
            normalized = np.clip(normalized, -1.0, 1.0)
        physical = (normalized + 1.0) * scale
        if modality == "ct":
            physical -= 1000.0
        if not np.isfinite(physical).all():
            raise ValueError(f"Decoded volume contains NaN or Inf for {case_dir.name}")
        nib.save(
            nib.Nifti1Image(physical.astype(np.float32, copy=False), reference.affine, reference.header),
            str(output_path),
        )
        outputs.append({"file": str(output_path), "shape": list(physical.shape),
                        "minimum": float(physical.min()), "maximum": float(physical.max())})
    return {"case": case_dir.name, "status": "decoded", "latent": str(latent_path),
            "reference": str(reference_path), "outputs": outputs}


def run(modality: str, config_name: str) -> None:
    args = parse_args(modality, config_name)
    for directory in (args.input_root, args.reference_root):
        if not directory.is_dir():
            raise FileNotFoundError(directory)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.gaussian_sigma <= 0:
        raise ValueError("gaussian_sigma must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu only for a very slow run.")
    if args.case_name:
        case_dirs = [args.input_root / args.case_name]
        if not case_dirs[0].is_dir():
            raise FileNotFoundError(case_dirs[0])
    else:
        case_dirs = sorted(path for path in args.input_root.iterdir() if path.is_dir())
    if not case_dirs:
        raise RuntimeError(f"No case directories found in {args.input_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if modality == "ct":
        config, decoder, codebook = load_ct_decoder(args.checkpoint, device)
    else:
        config, decoder, codebook = load_pet_decoder(args.checkpoint, device)
    print(f"{modality.upper()} VAE config: {config}")

    records = []
    for index, case_dir in enumerate(case_dirs, start=1):
        try:
            record = decode_case(case_dir, args.reference_root, args.output_root,
                                 decoder, codebook, device, args, modality)
        except Exception as exc:
            record = {"case": case_dir.name, "status": "failed", "reason": str(exc)}
        records.append(record)
        print(f"[{index}/{len(case_dirs)}] {case_dir.name}: {record['status']}")
        for output in record.get("outputs", []):
            print(f"  {output['file']} {output['shape']} range=[{output['minimum']:.1f}, {output['maximum']:.1f}]")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest = {"format": f"ct2pet_{modality}_latent_decode_manifest_v1",
                "input_root": str(args.input_root), "reference_root": str(args.reference_root),
                "output_root": str(args.output_root), "checkpoint": str(args.checkpoint),
                "latent_name": args.latent_name, "quantize": args.quantize,
                "clip_output": args.clip_output, "records": records}
    manifest_path = args.output_root / f"{modality}_decode_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved decode manifest to {manifest_path}")
    if any(record["status"] == "failed" for record in records):
        raise SystemExit(1)
