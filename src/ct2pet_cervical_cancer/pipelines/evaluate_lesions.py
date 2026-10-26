"""Evaluate manual lesions on a saved split, optionally generating CT-only PET first."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import nibabel as nib
import numpy as np

from ..inference.lesion_metrics import LesionMetricConfig, evaluate_case, summarize
from ..utils.cli import parse_config_args


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("split-manifest", "reference-root", "mask-root", "prediction-root", "output-dir",
                 "evaluation-mask-root", "checkpoint", "input-root", "pet-checkpoint"):
        parser.add_argument(f"--{name}", type=Path, default=None)
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--prediction-name", default="pet_prediction.nii.gz")
    parser.add_argument("--reference-clip-max", type=float, default=5.0)
    parser.add_argument("--hotspot-threshold", type=float, default=2.5)
    parser.add_argument("--match-iou", type=float, default=.1)
    parser.add_argument("--min-hotspot-volume-mm3", type=float, default=0.0)
    parser.add_argument("--connectivity", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-missing", action="store_true", help="Explicitly evaluate only existing cases and report omissions.")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate predictions; requires --checkpoint.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing metric reports, not predictions.")
    return parse_config_args(parser, "evaluate_lesions.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image(path: Path):
    image = nib.load(str(path))
    if len(image.shape) != 3 or not np.isfinite(image.affine).all():
        raise ValueError(f"Expected a finite-affine 3D NIfTI: {path}")
    # Current data use millimeters; unknown NIfTI units are explicitly reported.
    if image.header.get_xyzt_units()[0] not in ("mm", "unknown"):
        raise ValueError(f"Convert image geometry to millimeters before evaluation: {path}")
    return image


def check_grid(image, reference, path: Path):
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, rtol=1e-6, atol=1e-4):
        raise ValueError(f"NIfTI grid differs from reference PET; no implicit resampling: {path}")


class PETGenerator:
    """Load one checkpoint once; target PET and lesion masks never enter sampling."""
    def __init__(self, args):
        import torch
        from ..datasets.latent_transform import LatentTransform, checkpoint_identity
        from ..models.diffusion import CTControlDiffusion, CTControlDiffusionConfig, NoiseScheduler
        from ..models.vae.loading import load_pet_decoder
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; evaluate saved NIfTI without --checkpoint or use --device cpu")
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if payload.get("format") != "ct2pet_ct_control_checkpoint_v1":
            raise ValueError("Expected a checkpoint created by train_ct2pet.py")
        config = payload["configuration"]
        self.ct_transform = LatentTransform.from_dict(config["ct_transform"])
        self.pet_transform = LatentTransform.from_dict(config["pet_transform"])
        self.model = CTControlDiffusion(CTControlDiffusionConfig(**config["model_config"]))
        self.model.load_state_dict(payload[args.weights], strict=True)
        self.model.eval().to(self.device)
        self.scheduler = NoiseScheduler(**config["schedule_config"]).to(self.device)
        pet_checkpoint = args.pet_checkpoint or Path(config["arguments"]["pet_checkpoint"])
        if checkpoint_identity(str(pet_checkpoint))["sha256"] != self.pet_transform.pvae_checkpoint["sha256"]:
            raise ValueError("PET decoder checkpoint differs from training latent contract")
        _, self.decoder, self.codebook = load_pet_decoder(pet_checkpoint, self.device)
        self.input_root = args.input_root
        self.input_space = config["arguments"]["input_space"]
        self.quantize = bool(config["arguments"].get("decode_quantize", False))
        self.provenance = {"checkpoint_sha256": sha256(args.checkpoint),
                           "checkpoint_step": payload["global_step"], "weights": args.weights,
                           "seed": args.seed, "decoder_quantize": self.quantize,
                           "prediction_clipped": False}

    def generate(self, case: str, ct_image, prediction_path: Path, regenerate: bool):
        import torch
        from ..datasets.paired_latents import load_tensor, stable_seed
        from ..inference.decode import decode_sliding
        from ..inference.diffusion import sample_pet
        source = self.input_root / case / "ct.pt"
        provenance = {**self.provenance, "case": case, "ct_sha256": sha256(source),
                      "ct_shape": list(ct_image.shape), "ct_affine": ct_image.affine.tolist()}
        metadata_path = prediction_path.parent / "lesion_generation_metadata.json"
        if prediction_path.exists() and not regenerate:
            if not metadata_path.is_file():
                raise ValueError(f"Existing prediction has no matching generation metadata: {prediction_path}. "
                                 "Omit --checkpoint to evaluate saved images, or use --regenerate in a dedicated directory.")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if any(metadata.get(k) != value for k, value in provenance.items()) or metadata.get("prediction_sha256") != sha256(prediction_path):
                raise ValueError(f"Prediction provenance changed; select a new prediction-root: {prediction_path}")
            return provenance, "cached"
        with torch.inference_mode():
            ct = load_tensor(source)
            if self.input_space == "raw":
                ct = self.ct_transform.forward(ct)
            generated = sample_pet(self.model, self.scheduler, ct[None].to(self.device),
                                   stable_seed(self.provenance["seed"], case))[0]
            raw = self.pet_transform.inverse(generated)
            decoded = decode_sliding(self.decoder, raw[None], codebook=self.codebook,
                quantize=self.quantize, denormalize=False, window_size=(80, 80, 80),
                heart_size=(40, 40, 40), gaussian_sigma=.3)[0, 0].cpu().float().numpy()
        if decoded.shape != ct_image.shape or not np.isfinite(decoded).all():
            raise ValueError(f"Generated PET shape/values differ from reference CT: {case}")
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        header = ct_image.header.copy()
        header.set_data_dtype(np.float32)
        image = nib.Nifti1Image((decoded + 1) * 2.5, ct_image.affine, header)
        image.header.set_slope_inter(1, 0)
        nib.save(image, prediction_path)
        metadata = {**provenance, "prediction_sha256": sha256(prediction_path)}
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        return provenance, "generated"


def write_csv(path: Path, rows: list[dict]):
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["case"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args) -> dict:
    for key in ("split_manifest", "reference_root", "mask_root", "prediction_root", "output_dir"):
        if getattr(args, key) is None:
            raise ValueError(f"{key} is required")
    if Path(args.prediction_name).name != args.prediction_name:
        raise ValueError("prediction_name must be a file name, not a path")
    if args.regenerate and args.checkpoint is None:
        raise ValueError("--regenerate requires --checkpoint")
    if args.checkpoint is not None and args.input_root is None:
        raise ValueError("input_root is required when generating from a checkpoint")
    metric_config = LesionMetricConfig(args.reference_clip_max, args.hotspot_threshold, args.match_iou,
                                       args.min_hotspot_volume_mm3, args.connectivity)
    payload = json.loads(args.split_manifest.read_text(encoding="utf-8-sig"))
    records = payload.get("records", [])
    seen = set()
    selected = []
    for row in records:
        case = row.get("case")
        if not isinstance(case, str) or not case or case in (".", "..") or "/" in case or "\\" in case or case in seen:
            raise ValueError("Split manifest must contain unique, safe case names")
        if row.get("split") not in ("train", "validation", "test"):
            raise ValueError(f"Unknown split for {case}")
        seen.add(case)
        if row["split"] == args.split:
            selected.append(case)
    if not selected:
        raise ValueError(f"No {args.split} cases in split manifest")
    reports = ("summary.json", "per_case.csv", "per_lesion.csv", "per_hotspot.csv", "results.json")
    if not args.overwrite and any((args.output_dir / name).exists() for name in reports):
        raise FileExistsError("Evaluation report exists; select a new --output-dir or use --overwrite")
    # Full-cohort input checks precede model loading or report creation.
    plans, skipped = [], []
    for case in selected:
        reference_path = args.reference_root / case / "pet.nii.gz"
        mask_path = args.mask_root / f"{case}.nii.gz"
        prediction_path = args.prediction_root / case / args.prediction_name
        evaluation_path = args.evaluation_mask_root / f"{case}.nii.gz" if args.evaluation_mask_root else None
        protected = [reference_path, mask_path, args.reference_root / case / "ct.nii.gz"]
        if evaluation_path:
            protected.append(evaluation_path)
        if prediction_path.resolve() in {p.resolve() for p in protected}:
            raise ValueError(f"Prediction path must not alias a reference image or annotation: {prediction_path}")
        required = [reference_path, mask_path] + ([evaluation_path] if evaluation_path else [])
        if args.checkpoint is None:
            required.append(prediction_path)
        else:
            required.append(args.reference_root / case / "ct.nii.gz")
            if args.input_root is not None:
                required.append(args.input_root / case / "ct.pt")
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            if not args.allow_missing:
                raise FileNotFoundError(f"Missing input for {case}: {missing}. Default evaluation requires the full split.")
            skipped.append({"case": case, "missing": missing})
            continue
        reference = read_image(reference_path)
        check_grid(read_image(mask_path), reference, mask_path)
        if evaluation_path:
            check_grid(read_image(evaluation_path), reference, evaluation_path)
        if args.checkpoint is None:
            check_grid(read_image(prediction_path), reference, prediction_path)
        else:
            ct_path = args.reference_root / case / "ct.nii.gz"
            check_grid(read_image(ct_path), reference, ct_path)
        plans.append((case, reference_path, mask_path, prediction_path, evaluation_path))
    if not plans:
        raise ValueError("No evaluable cases remain")
    generator = PETGenerator(args) if args.checkpoint is not None else None
    cases, lesions, hotspots = [], [], []
    for index, (case, reference_path, mask_path, prediction_path, evaluation_path) in enumerate(plans, 1):
        print(f"[{index}/{len(plans)}] {case}", flush=True)
        reference = read_image(reference_path)
        if generator:
            ct_image = read_image(args.reference_root / case / "ct.nii.gz")
            generator.generate(case, ct_image, prediction_path, args.regenerate)
        image, mask = read_image(prediction_path), read_image(mask_path)
        check_grid(image, reference, prediction_path)
        values = np.asanyarray(image.dataobj).astype(np.float32)
        target = np.asanyarray(reference.dataobj).astype(np.float32)
        annotation = np.asanyarray(mask.dataobj)
        region = np.asanyarray(read_image(evaluation_path).dataobj) if evaluation_path else None
        row, lesion_rows, hotspot_rows = evaluate_case(values, target, annotation, reference.affine, metric_config, region)
        row.update(case=case, prediction=str(prediction_path), reference=str(reference_path), mask=str(mask_path),
                   spatial_units=reference.header.get_xyzt_units()[0],
                   prediction_sha256=sha256(prediction_path), reference_sha256=sha256(reference_path),
                   mask_sha256=sha256(mask_path), evaluation_mask_sha256=sha256(evaluation_path) if evaluation_path else None)
        cases.append(row)
        lesions.extend(dict(case=case, **r) for r in lesion_rows)
        hotspots.extend(dict(case=case, **r) for r in hotspot_rows)
    summary = summarize(cases, lesions)
    summary.update(requested_cases=len(selected), skipped_cases=len(skipped), complete_split=not skipped)
    protocol = {"split": args.split, "split_manifest_sha256": sha256(args.split_manifest),
                "case_names": [row["case"] for row in cases], "metric_config": asdict(metric_config),
                "generation": generator.provenance if generator else None,
                "prediction_clipped_by_evaluator": False,
                "matching": "one-to-one; maximize eligible IoU matches, then total IoU",
                "notes": ["Hotspots are thresholded uptake, not semantic tumor segmentations.",
                          "Unmatched hotspots may be physiological uptake; use a predefined evaluation mask for exclusions.",
                          "Unknown NIfTI spatial units are assumed to be millimeters.",
                          "Default intensity errors use reference clipped to [0,5]; raw_* metrics use original reference values.",
                          "Hotspot threshold and IoU settings are exploratory and must be fixed before comparing test results."]}
    result = {"summary": summary, "protocol": protocol, "skipped": skipped,
              "cases": cases, "lesions": lesions, "hotspots": hotspots}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_case.csv", cases)
    write_csv(args.output_dir / "per_lesion.csv", lesions)
    write_csv(args.output_dir / "per_hotspot.csv", hotspots)
    for name, content in (("summary.json", {"summary": summary, "protocol": protocol, "skipped": skipped}),
                          ("results.json", result)):
        (args.output_dir / name).write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return result


def main():
    evaluate(parse_args())
