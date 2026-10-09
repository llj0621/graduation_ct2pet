"""Fixed-case CT-only sampling and evaluation against actual paired PET NIfTI."""
import json
import math
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from ..datasets.paired_latents import stable_seed
from ..inference.decode import decode_sliding
from ..inference.diffusion import sample_pet
from ..models.vae.loading import load_pet_decoder


def pet_volume_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Generated and reference PET must share a 3D voxel grid")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("PET metrics require finite volumes")
    difference = prediction.float() - target.float()
    mse = difference.square().mean().item()
    # null PSNR records exact equality without serializing Infinity into JSON/W&B.
    values = {"mae": difference.abs().mean().item(), "mse": mse,
              "psnr": 10 * math.log10(25 / mse) if mse > 0 else None}
    from monai.metrics import SSIMMetric
    metric = SSIMMetric(spatial_dims=2, data_range=5.0, win_size=7)
    axial_pred = prediction.permute(2, 0, 1)[:, None]
    axial_target = target.permute(2, 0, 1)[:, None]
    if min(axial_pred.shape[-2:]) < 7:
        raise ValueError("PET axial dimensions must be >=7 for SSIM")
    total, count = 0.0, 0
    for start in range(0, axial_pred.shape[0], 8):
        scores = metric(axial_pred[start:start + 8], axial_target[start:start + 8])
        total += float(scores.sum())
        count += scores.numel()
        metric.reset()
    values["ssim_axial"] = total / count
    return values


def preview_images(prediction: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    panels = {}
    for axis, name in enumerate(("sagittal", "coronal", "axial")):
        generated = np.take(prediction, prediction.shape[axis] // 2, axis=axis)
        actual = np.take(target, target.shape[axis] // 2, axis=axis)
        # Left: actual PET, middle: generated PET, right: absolute error, fixed [0,5] scale.
        panels[name] = (np.clip(np.concatenate((actual, generated, np.abs(actual - generated)), axis=1), 0, 5)
                        / 5 * 255).astype(np.uint8)
    return panels


@torch.inference_mode()
def evaluate_generated(model, scheduler, dataset, output_dir: Path, seed: int,
                       count: int, device: torch.device, reference_root: Path | None,
                       pet_checkpoint: Path, quantize: bool, logger, prefix: str,
                       step: int) -> dict:
    indices = sorted(range(len(dataset)), key=lambda index: stable_seed(seed, dataset.records[index]["case"]))[:count]
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    decoder = codebook = None
    try:
        for index in indices:
            item = dataset[index]
            case = item["case"]
            ct = item["ct"].unsqueeze(0).to(device)
            # Only CT is passed to sampling; target PET is used below for metrics.
            generated = sample_pet(model, scheduler, ct, stable_seed(seed, case)).cpu()[0]
            raw = dataset.pet_transform.inverse(generated)
            case_dir = output_dir / case
            case_dir.mkdir(parents=True, exist_ok=True)
            torch.save(raw, case_dir / "pet_pred.pt")
            difference = generated - item["pet"]
            row = {"case": case, "seed": stable_seed(seed, case),
                   "latent_mse": difference.square().mean().item(),
                   "latent_mae": difference.abs().mean().item(),
                   "pet_latent": str(case_dir / "pet_pred.pt"), "latent_space": "raw",
                   "decoder_quantize": quantize}
            if reference_root is not None:
                reference_path = reference_root / case / "pet.nii.gz"
                reference = nib.load(str(reference_path))
                if decoder is None:
                    _, decoder, codebook = load_pet_decoder(pet_checkpoint, device)
                decoded = decode_sliding(
                    decoder=decoder, input=raw.unsqueeze(0).to(device), codebook=codebook,
                    quantize=quantize, denormalize=False, window_size=(80, 80, 80),
                    heart_size=(40, 40, 40), gaussian_sigma=0.3,
                )[0, 0].cpu().float()
                physical = (decoded + 1) * 2.5
                actual = torch.from_numpy(np.asanyarray(reference.dataobj).astype(np.float32)).clamp(0, 5)
                row.update(pet_volume_metrics(physical, actual))
                row.update(reference=str(reference_path), reference_source="paired_nifti_clipped_0_5",
                           metric_range=5.0)
                image_path = case_dir / "pet_prediction.nii.gz"
                nib.save(nib.Nifti1Image(physical.numpy(), reference.affine, reference.header), image_path)
                row["pet_prediction"] = str(image_path)
                logger.images({f"{prefix}/images/{case}/{name}": image
                               for name, image in preview_images(physical.numpy(), actual.numpy()).items()}, step)
            (case_dir / "prediction_metadata.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            records.append(row)
    finally:
        del decoder, codebook
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (output_dir / "per_case_metrics.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    summary = {f"{prefix}/generated_count": len(records)}
    for name in ("latent_mse", "latent_mae", "mae", "mse", "psnr", "ssim_axial"):
        values = [row[name] for row in records if row.get(name) is not None]
        if values:
            summary[f"{prefix}/generated_{name}"] = float(np.mean(values))
    if logger.run is not None and records:
        import wandb
        columns = ["case", "latent_mse", "latent_mae", "mae", "mse", "psnr", "ssim_axial"]
        logger.run.log({f"{prefix}/cases": wandb.Table(columns=columns,
                       data=[[row.get(name) for name in columns] for row in records]), "global_step": step})
    logger.log(summary, step)
    return summary
