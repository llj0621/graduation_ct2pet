"""Generate PET from a CT latent without loading a target PET."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from ..datasets.latent_transform import LatentTransform, checkpoint_identity
from ..datasets.paired_latents import load_tensor, stable_seed
from ..inference.decode import decode_sliding
from ..inference.diffusion import sample_pet
from ..models.diffusion import CTControlDiffusion, CTControlDiffusionConfig, NoiseScheduler
from ..models.vae.loading import load_pet_decoder
from ..utils.cli import parse_config_args


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "reference-root", "output-root", "checkpoint", "pet-checkpoint"):
        parser.add_argument(f"--{name}", type=Path, default=None)
    parser.add_argument("--latent-name", default="ct.pt")
    parser.add_argument("--input-space", choices=("raw", "codebook_minmax"), default="raw")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--case", dest="case_name", default=None)
    parser.add_argument("--decode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quantize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clip-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parse_config_args(parser, "infer_ct2pet.json")


@torch.inference_mode()
def infer(args):
    if args.input_root is None or not args.input_root.is_dir():
        raise FileNotFoundError(args.input_root)
    if args.checkpoint is None or not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output_root is None:
        raise ValueError("output_root is required")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != "ct2pet_ct_control_checkpoint_v1":
        raise ValueError("Expected a checkpoint created by train_ct2pet.py")
    configuration = payload["configuration"]
    ct_transform = LatentTransform.from_dict(configuration["ct_transform"])
    pet_transform = LatentTransform.from_dict(configuration["pet_transform"])
    model = CTControlDiffusion(CTControlDiffusionConfig(**configuration["model_config"]))
    model.load_state_dict(payload[args.weights], strict=True)
    model.eval().to(device)
    scheduler = NoiseScheduler(**configuration["schedule_config"]).to(device)
    checkpoint_step = payload["global_step"]
    del payload
    decoder = codebook = None
    if args.decode:
        if args.reference_root is None or not args.reference_root.is_dir():
            raise FileNotFoundError(args.reference_root)
        pet_checkpoint = args.pet_checkpoint or Path(configuration["arguments"]["pet_checkpoint"])
        if checkpoint_identity(str(pet_checkpoint))["sha256"] != pet_transform.pvae_checkpoint["sha256"]:
            raise ValueError("PET decoder checkpoint differs from the training latent contract")
        _, decoder, codebook = load_pet_decoder(pet_checkpoint, device)
    cases = ([args.input_root / args.case_name] if args.case_name else
             sorted(path for path in args.input_root.iterdir() if path.is_dir() and (path / args.latent_name).is_file()))
    if not cases:
        raise ValueError("No CT latent case directories found")
    records = []
    for case_dir in cases:
        output = args.output_root / case_dir.name
        if not args.overwrite and any((output / name).exists() for name in
                                     ("pet_pred.pt", "pet_prediction.nii.gz", "prediction_metadata.json")):
            raise FileExistsError(f"Output exists for {case_dir.name}; use --overwrite")
        ct = load_tensor(case_dir / args.latent_name)
        if args.input_space == "raw":
            ct = ct_transform.forward(ct)
        generated = sample_pet(model, scheduler, ct.unsqueeze(0).to(device),
                               stable_seed(args.seed, case_dir.name))[0].cpu()
        raw = pet_transform.inverse(generated)
        reference = None
        physical = None
        if args.decode:
            # Inference inherits geometry from CT; it does not open any PET input.
            reference = nib.load(str(args.reference_root / case_dir.name / "ct.nii.gz"))
            decoded = decode_sliding(decoder, raw.unsqueeze(0).to(device), codebook=codebook,
                quantize=args.quantize, denormalize=False, window_size=(80, 80, 80),
                heart_size=(40, 40, 40), gaussian_sigma=0.3)[0, 0].cpu().float().numpy()
            if tuple(decoded.shape) != tuple(reference.shape):
                raise ValueError(f"Decoded shape {decoded.shape} differs from reference CT {reference.shape}")
            physical = (decoded + 1) * 2.5
            if args.clip_output:
                physical = np.clip(physical, 0, 5)
            if not np.isfinite(physical).all():
                raise ValueError("PET decoder produced NaN or Inf")
        output.mkdir(parents=True, exist_ok=True)
        torch.save(raw, output / "pet_pred.pt")
        if physical is not None:
            nib.save(nib.Nifti1Image(physical, reference.affine, reference.header), output / "pet_prediction.nii.gz")
        row = {"case": case_dir.name, "ct_latent": str(case_dir / args.latent_name),
               "pet_latent": str(output / "pet_pred.pt"), "latent_space": "raw",
               "latent_shape": list(raw.shape), "checkpoint": str(args.checkpoint),
               "checkpoint_step": checkpoint_step, "weights": args.weights,
               "seed": stable_seed(args.seed, case_dir.name), "decoder_quantize": args.quantize,
               "clip_output": args.clip_output, "pet_transform": pet_transform.to_dict()}
        (output / "prediction_metadata.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        records.append(row)
        print(f"{case_dir.name}: saved raw PET latent {list(raw.shape)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "inference_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def main():
    infer(parse_args())
