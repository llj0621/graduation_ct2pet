"""CT/PET reconstruction using the existing PVAE checkpoints."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import torch
from ..models.vae.loading import load_pvae
from ..datasets.nifti import prepare_ct, prepare_pet, save_outputs
from ..inference.encode import encode_sliding
from ..inference.decode import decode_sliding

from ..utils.cli import parse_config_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=None)
    parser.add_argument(
        "--ct-checkpoint", type=Path, default=None
    )
    parser.add_argument(
        "--pet-checkpoint", type=Path, default=None
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--encode-stride",
        type=int,
        default=40,
        choices=(40, 80),
        help="Image-space sliding-window stride; 40 matches the project's default.",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=0.3,
        help="Sigma for Gaussian blending in both encoding and decoding.",
    )
    return parse_config_args(parser, "reconstruct_vae.json")


@torch.inference_mode()
def reconstruct(image, encoder, decoder, codebook, encode_stride, gaussian_sigma):
    latent = encode_sliding(
        image,
        encoder,
        codebook=codebook,
        quantize=False,
        normalize=False,
        gaussian_sigma=gaussian_sigma,
        heart_size=(encode_stride,) * 3,
    )
    return decode_sliding(
        decoder,
        latent,
        codebook=codebook,
        quantize=True,
        denormalize=False,
        gaussian_sigma=gaussian_sigma,
        window_size=(80, 80, 80),
        heart_size=(40, 40, 40),
    )


def main() -> None:
    args = parse_args()
    ct_path, pet_path = args.case_dir / "ct.nii.gz", args.case_dir / "pet.nii.gz"
    for path in (ct_path, pet_path, args.ct_checkpoint, args.pet_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu only for a very slow test.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cases = (
        ("ct", prepare_ct(ct_path), args.ct_checkpoint),
        ("pet", prepare_pet(pet_path, ct_path), args.pet_checkpoint),
    )
    for modality, (image, affine), checkpoint in cases:
        config, encoder, decoder, codebook = load_pvae(checkpoint, modality, device)
        print(f"Running {modality.upper()} PVAE with {asdict(config)}")
        reconstruction = reconstruct(
            image.to(device),
            encoder,
            decoder,
            codebook,
            args.encode_stride,
            args.gaussian_sigma,
        )
        save_outputs(reconstruction, affine, modality, args.output_dir)
        print(f"Saved {modality} reconstruction to {args.output_dir}")
        del encoder, decoder, codebook, reconstruction
        torch.cuda.empty_cache()
