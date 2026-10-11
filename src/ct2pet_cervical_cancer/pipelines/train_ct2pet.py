"""Train the CT-only B1 baseline on this project's paired continuous latents."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from ..datasets.latent_transform import LatentTransform
from ..datasets.paired_latents import (PairedLatentDataset, ShapeBatchSampler, DistributedShapeBatchSampler, checkpoint_transform,
                                      discover_pairs, resolve_split, stable_seed, file_sha256)
from ..inference.evaluate_ct2pet import evaluate_generated
from ..models.diffusion import CTControlDiffusion, CTControlDiffusionConfig, NoiseScheduler
from ..models.diffusion.losses import IMAGE_L1_DEFAULTS, ImageL1Loss
from ..utils.cli import parse_config_args
from ..utils.experiment import ExperimentLogger, save_checkpoint
from ..utils.distributed import DistributedContext


def int_tuple(value) -> tuple[int, ...]:
    return tuple(int(item) for item in (value.split(",") if isinstance(value, str) else value))


def write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "split-manifest", "patient-map", "ct-checkpoint", "pet-checkpoint",
                 "output-dir", "reference-root", "resume"):
        parser.add_argument(f"--{name}", type=Path, default=None)
    parser.add_argument("--input-space", choices=("raw", "codebook_minmax"), default="raw")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dist-backend", choices=("auto", "nccl", "gloo"), default="auto")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="float32")
    for name, default in (("seed", 42), ("batch-size", 8), ("num-workers", 0), ("max-steps", 300000),
                          ("base-channels", 64), ("num-res-blocks", 2), ("time-dim", 256),
                          ("num-heads", 4), ("num-timesteps", 1000), ("log-interval", 50),
                          ("validation-interval", 2000), ("checkpoint-interval", 20000),
                          ("sample-interval", 20000), ("validation-samples", 2), ("test-samples", 4)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--channel-mult", type=int_tuple, default=(1, 2, 4, 8))
    parser.add_argument("--attention-levels", type=int_tuple, default=(2, 3))
    for name, default in (("learning-rate", 1e-4), ("weight-decay", 0.01), ("ema-decay", 0.9999),
                          ("min-snr-gamma", 5.0), ("grad-clip", 1.0),
                          ("validation-fraction", 0.1), ("test-fraction", 0.1)):
        parser.add_argument(f"--{name}", type=float, default=default)
    parser.add_argument("--decode-quantize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--image-l1-weight", type=float, default=0.0,
                        help="Weight of decoded PET patch L1; zero disables decoder loading and image loss.")
    parser.add_argument("--image-l1-target", choices=("vae_reconstruction", "paired_nifti"), default="vae_reconstruction")
    parser.add_argument("--image-l1-timestep-weighting", choices=("none", "linear"), default="linear")
    parser.add_argument("--image-l1-patch-size", type=int, default=80, help="Decoded image patch edge length in voxels.")
    parser.add_argument("--wandb-project", default="ct2pet-cervical-cancer")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--run-name", default="ct-control-baseline")
    parser.add_argument("--stop-after", type=int, default=None,
                        help="Stop after this global step while retaining the original max_steps schedule.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and save the split/contract without training or W&B.")
    return parse_config_args(parser, "train_ct2pet.json")


def validate_args(args):
    if any(getattr(args, name) is None for name in
           ("input_root", "split_manifest", "ct_checkpoint", "pet_checkpoint", "output_dir")):
        raise ValueError("Input root, split manifest, VAE checkpoints and output directory are required")
    if args.reference_root is not None and not args.reference_root.is_dir():
        raise FileNotFoundError(args.reference_root)
    if any(getattr(args, name) < 1 for name in ("batch_size", "max_steps", "log_interval",
           "validation_interval", "checkpoint_interval", "sample_interval")):
        raise ValueError("Batch size, steps and logging/checkpoint intervals must be positive")
    if args.num_workers < 0 or args.validation_samples < 0 or args.test_samples < 0:
        raise ValueError("Workers and generated-sample counts cannot be negative")
    if (not 0 <= args.ema_decay < 1 or args.learning_rate <= 0 or args.weight_decay < 0
            or args.grad_clip <= 0):
        raise ValueError("Invalid learning rate, weight decay, EMA decay or gradient clipping")
    if args.stop_after is not None and not 1 <= args.stop_after <= args.max_steps:
        raise ValueError("stop_after must be between one and max_steps")
    if not math.isfinite(args.image_l1_weight) or args.image_l1_weight < 0:
        raise ValueError("image_l1_weight must be finite and nonnegative")
    if args.image_l1_patch_size < 1:
        raise ValueError("image_l1_patch_size must be positive")
    if args.image_l1_target not in {"vae_reconstruction", "paired_nifti"}:
        raise ValueError("Unknown image_l1_target")
    if args.image_l1_timestep_weighting not in {"none", "linear"}:
        raise ValueError("Unknown image_l1_timestep_weighting")
    if args.image_l1_weight > 0 and args.image_l1_target == "paired_nifti" and args.reference_root is None:
        raise ValueError("paired_nifti image L1 requires reference_root")
    if args.resume is None and any((args.output_dir / name).exists() for name in ("config.json", "last.pt", "metrics.jsonl")):
        raise FileExistsError("Experiment already exists; choose another --output-dir or use --resume last.pt")
    if args.resume is not None and args.resume.resolve().parent != args.output_dir.resolve():
        raise ValueError("Resume checkpoint must belong to --output-dir, preserving its best checkpoint and history")


def make_loader(dataset, batch_size: int, workers: int, shuffle: bool, seed: int,
                pin_memory: bool = False, context: DistributedContext | None = None, training: bool = False):
    sampler = (DistributedShapeBatchSampler(dataset.records, batch_size, shuffle, seed,
               context.rank, context.world_size, training) if context and context.enabled else
               ShapeBatchSampler(dataset.records, batch_size, shuffle, seed))
    return DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                      pin_memory=pin_memory, generator=torch.Generator().manual_seed(seed)), sampler


@torch.inference_mode()
def validate(model, scheduler, loader, device, seed, context: DistributedContext | None = None,
             image_loss: ImageL1Loss | None = None, image_l1_weight: float = 0.0) -> dict:
    was_training = model.training
    model.eval()
    total_loss = total_mse = total_mae = count = 0
    total_image_l1 = total_image_weighted = 0.0
    try:
        for batch in loader:
            ct, pet = batch["ct"].to(device), batch["pet"].to(device)
            noise, timesteps = [], []
            for case, sample in zip(batch["case"], pet, strict=True):
                generator = torch.Generator(device=device).manual_seed(stable_seed(seed, case))
                timesteps.append(torch.randint(scheduler.num_timesteps, (1,), device=device, generator=generator))
                noise.append(torch.randn(sample.shape, device=device, generator=generator))
            timesteps, noise = torch.cat(timesteps), torch.stack(noise)
            noisy = scheduler.add_noise(pet, noise, timesteps)
            prediction = model(noisy, timesteps, ct)
            loss = scheduler.loss(prediction, scheduler.velocity(pet, noise, timesteps), timesteps)
            clean_prediction = scheduler.predict_x0(noisy, prediction, timesteps)
            difference = clean_prediction - pet
            batch_size = len(pet)
            total_loss += float(loss) * batch_size
            total_mse += float(difference.square().flatten(1).mean(1).sum())
            total_mae += float(difference.abs().flatten(1).mean(1).sum())
            if image_loss is not None:
                image_metrics = image_loss(clean_prediction, pet, timesteps, scheduler.num_timesteps, batch["case"], seed)
                total_image_l1 += float(image_metrics["image_l1"]) * batch_size
                total_image_weighted += float(image_metrics["image_l1_weighted"]) * batch_size
            count += batch_size
        if context is not None:
            total_loss, total_mse, total_mae, count = context.sum([total_loss, total_mse, total_mae, count])
            if image_loss is not None:
                total_image_l1, total_image_weighted = context.sum([total_image_l1, total_image_weighted])
        if count == 0:
            raise ValueError("Validation/test split contains no samples")
        result = {"v_loss": total_loss / count, "x0_mse": total_mse / count,
                  "x0_mae": total_mae / count, "case_count": int(count)}
        if image_loss is not None:
            result.update(image_l1=total_image_l1 / count, image_l1_weighted=total_image_weighted / count,
                          total_loss=(total_loss + image_l1_weight * total_image_weighted) / count)
        if not all(np.isfinite(value) for value in result.values()):
            raise ValueError("Validation produced non-finite metrics")
        return result
    finally:
        model.train(was_training)


def train(args) -> dict:
    # Programmatic callers and old configuration files retain the baseline defaults.
    for key, default in IMAGE_L1_DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, default)
    context = DistributedContext.from_environment(args.device, getattr(args, "dist_backend", "auto"))
    try:
        return train_with_context(args, context)
    finally:
        context.close()


def prepare_inputs(args):
    validate_args(args)
    records = discover_pairs(args.input_root)
    split = resolve_split(records, args.split_manifest, args.seed, args.validation_fraction,
                          args.test_fraction, args.patient_map)
    ct_transform, pet_transform = checkpoint_transform(args.ct_checkpoint), checkpoint_transform(args.pet_checkpoint)
    if args.image_l1_weight > 0:
        from ..models.vae.loading import load_config
        from ..models.vae.pet import PETEncoderDecoderConfig
        config = load_config(args.pet_checkpoint, PETEncoderDecoderConfig)
        upscale = 8 if config.scale == "8X" else 4
        if args.image_l1_patch_size % upscale:
            raise ValueError("image_l1_patch_size must be a multiple of the PET decoder scale")
        if args.image_l1_target == "paired_nifti":
            import nibabel as nib
            for row in split["records"]:
                path = args.reference_root / row["case"] / "pet.nii.gz"
                expected = tuple(size * upscale for size in row["shape"][1:])
                if nib.load(str(path)).shape != expected:
                    raise ValueError(f"PET reference grid differs for {row['case']}; expected {expected}")
                # Include the image supervision data in the exact-resume contract.
                row["image_l1_reference_sha256"] = file_sha256(path)
    return records, split, ct_transform.to_dict(), pet_transform.to_dict()


def train_with_context(args, context) -> dict:
    device = context.device
    if device.type != "cuda" and args.precision != "float32":
        raise ValueError("Mixed precision requires CUDA")
    if args.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("The selected CUDA device does not support bfloat16")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    records, split, ct_values, pet_values = context.on_main(lambda: prepare_inputs(args))
    ct_transform, pet_transform = LatentTransform.from_dict(ct_values), LatentTransform.from_dict(pet_values)
    model_config = CTControlDiffusionConfig(base_channels=args.base_channels, channel_mult=args.channel_mult,
        num_res_blocks=args.num_res_blocks, time_dim=args.time_dim,
        attention_levels=args.attention_levels, num_heads=args.num_heads)
    schedule_config = {"num_timesteps": args.num_timesteps, "min_snr_gamma": args.min_snr_gamma}
    datasets = {name: PairedLatentDataset([row for row in split["records"] if row["split"] == name],
                 ct_transform, pet_transform, args.input_space) for name in ("train", "validation", "test")}
    identity = hashlib.sha256(json.dumps(split["records"], sort_keys=True).encode()).hexdigest()
    configuration = {"format": "ct2pet_ct_control_training_v1",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model_config": asdict(model_config), "schedule_config": schedule_config,
        "loss_config": {"primary": "min_snr_velocity_mse", "image_l1_weight": args.image_l1_weight,
            "image_l1_target": args.image_l1_target, "image_l1_timestep_weighting": args.image_l1_timestep_weighting,
            "image_l1_patch_size": args.image_l1_patch_size, "decoder_frozen": True,
            "decoder_quantize": False, "prediction_clipped": False, "image_coordinates": "decoder_normalized"},
        "ct_transform": ct_transform.to_dict(), "pet_transform": pet_transform.to_dict(),
        "data_identity": identity, "split_grouping": split.get("grouping", "explicit_manifest"),
        "split_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "shape_counts": {str(shape): sum(tuple(row["shape"]) == shape for row in records)
                         for shape in sorted({tuple(row["shape"]) for row in records})},
        "prediction_type": "velocity", "beta_schedule": "linear", "sampling": "DDPM_beta_variance",
        "conditioning": "ct_only", "source": "Leaper Phase B1 CT-control baseline",
        "distributed": {"world_size": context.world_size, "backend": context.backend,
            "device_type": device.type, "batch_size_per_rank": args.batch_size,
            "global_batch_size_limit": args.batch_size * context.world_size,
            "training_tail_policy": "repeat_only_if_smaller_than_world_size",
            "evaluation_padding": False}}
    context.on_main(lambda: args.output_dir.mkdir(parents=True, exist_ok=True))
    if context.is_main:
        print(json.dumps({key: configuration[key] for key in
                         ("split_counts", "split_grouping", "shape_counts", "distributed")}, indent=2))
    if args.dry_run:
        NoiseScheduler(**schedule_config)
        context.on_main(lambda: write_json(args.output_dir / "dry_run_report.json", configuration))
        return configuration
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if checkpoint:
        if checkpoint.get("format") != "ct2pet_ct_control_checkpoint_v1":
            raise ValueError("Expected a checkpoint created by this CT2PET trainer")
        old = checkpoint["configuration"]
        previous_world = old.get("distributed", {}).get("world_size", 1)
        if previous_world != context.world_size:
            raise ValueError("Resume world size differs; exact resume requires the original number of processes")
        previous_device = old.get("distributed", {}).get("device_type", torch.device(old["arguments"]["device"]).type)
        if previous_device != device.type:
            raise ValueError("Exact resume requires the original device type")
        if context.enabled and old.get("distributed", {}).get("backend") != context.backend:
            raise ValueError("Exact DDP resume requires the original communication backend")
        if context.enabled and len(checkpoint.get("rng_states", [])) != context.world_size:
            raise ValueError("DDP checkpoint must contain RNG states for every rank")
        for key in ("model_config", "schedule_config", "data_identity", "ct_transform", "pet_transform"):
            if json.dumps(old[key], sort_keys=True) != json.dumps(configuration[key], sort_keys=True):
                raise ValueError(f"Resume configuration differs: {key}")
        for key in ("input_space", "batch_size", "seed", "max_steps", "precision", "learning_rate", "weight_decay", "ema_decay", "grad_clip"):
            if old["arguments"][key] != configuration["arguments"][key]:
                raise ValueError(f"Resume training setting differs: {key}")
        if old["arguments"].get("image_l1_weight", 0.0) != args.image_l1_weight:
            raise ValueError("Resume training setting differs: image_l1_weight; start a separate experiment")
        if args.image_l1_weight > 0:
            for key, default in IMAGE_L1_DEFAULTS.items():
                if old["arguments"].get(key, default) != configuration["arguments"][key]:
                    raise ValueError(f"Resume training setting differs: {key}")
            if args.image_l1_target == "paired_nifti" and old["arguments"].get("reference_root") != str(args.reference_root):
                raise ValueError("Resume image L1 reference_root differs")
    model = CTControlDiffusion(model_config).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    training_model = (DistributedDataParallel(model,
        device_ids=[device.index] if device.type == "cuda" else None,
        output_device=device.index if device.type == "cuda" else None,
        broadcast_buffers=False) if context.enabled else model)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    scheduler = NoiseScheduler(**schedule_config).to(device)
    image_loss = (ImageL1Loss.from_checkpoint(args.pet_checkpoint, pet_transform, device,
        patch_size=args.image_l1_patch_size, target=args.image_l1_target,
        timestep_weighting=args.image_l1_timestep_weighting, reference_root=args.reference_root)
        if args.image_l1_weight > 0 else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    learning_schedule = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.max_steps, power=1)
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "float16")
    step = epoch = batch_offset = 0
    best_loss = float("inf")
    if checkpoint:
        ema.load_state_dict(checkpoint["ema"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        learning_schedule.load_state_dict(checkpoint["learning_schedule"])
        scaler.load_state_dict(checkpoint["scaler"])
        step, epoch, batch_offset = checkpoint["global_step"], checkpoint["epoch"], checkpoint["batch_offset"]
        best_loss = checkpoint["best_validation_loss"]
    train_loader, sampler = make_loader(datasets["train"], args.batch_size, args.num_workers, True, args.seed,
                                       device.type == "cuda", context, training=True)
    val_loader, _ = make_loader(datasets["validation"], args.batch_size, args.num_workers, False, args.seed,
                               device.type == "cuda", context)
    test_loader, _ = make_loader(datasets["test"], args.batch_size, args.num_workers, False, args.seed,
                                device.type == "cuda", context)
    configuration["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    context.on_main(lambda: write_json(args.output_dir / "config.json", configuration))
    logger = None

    def start_logger():
        nonlocal logger
        logger = ExperimentLogger(args.output_dir, configuration, args.wandb_project, args.wandb_entity,
                                  args.wandb_mode, args.run_name,
                                  checkpoint.get("wandb_run_id") if checkpoint else None, bool(checkpoint))
        return logger.run_id

    if checkpoint:
        context.restore_rng(checkpoint)
    elif context.enabled:
        # Identical initial weights, independent training noise on each rank.
        random.seed(args.seed + context.rank)
        np.random.seed(args.seed + context.rank)
        torch.manual_seed(args.seed + context.rank)
    # Logging startup must not alter the saved per-rank training RNG stream.
    states_before_logging = context.rng_states()
    run_id = context.on_main(start_logger)
    context.restore_rng({"rng_states": states_before_logging})
    if checkpoint:
        del checkpoint

    def payload(rng_states):
        return {"format": "ct2pet_ct_control_checkpoint_v1", "configuration": configuration,
            "model": model.state_dict(), "ema": ema.state_dict(), "optimizer": optimizer.state_dict(),
            "learning_schedule": learning_schedule.state_dict(), "scaler": scaler.state_dict(),
            "global_step": step, "epoch": epoch, "batch_offset": batch_offset,
            "best_validation_loss": best_loss, "wandb_run_id": run_id, "rng_states": rng_states,
            "torch_rng": rng_states[0]["torch_rng"], "python_rng": rng_states[0]["python_rng"],
            "numpy_rng": rng_states[0]["numpy_rng"],
            "cuda_rng": [rng_states[0]["cuda_rng"]] if device.type == "cuda" else []}

    def save(paths):
        states = context.rng_states()
        def write():
            for path in paths:
                save_checkpoint(payload(states), path)
        context.on_main(write)

    failed = True
    try:
        target_step = args.stop_after or args.max_steps
        while step < target_step:
            sampler.epoch = epoch
            for index, batch in enumerate(train_loader):
                if index < batch_offset:
                    continue
                training_model.train()
                optimizer.zero_grad(set_to_none=True)
                ct, pet = batch["ct"].to(device), batch["pet"].to(device)
                timesteps = torch.randint(scheduler.num_timesteps, (pet.shape[0],), device=device)
                noise = torch.randn_like(pet)
                noisy = scheduler.add_noise(pet, noise, timesteps)
                dtype = torch.float16 if args.precision == "float16" else torch.bfloat16
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=args.precision != "float32"):
                    prediction = training_model(noisy, timesteps, ct)
                    v_loss = scheduler.loss(prediction, scheduler.velocity(pet, noise, timesteps), timesteps)
                loss = v_loss
                image_metrics = {}
                if image_loss is not None:
                    image_metrics = image_loss(scheduler.predict_x0(noisy, prediction, timesteps), pet,
                                               timesteps, scheduler.num_timesteps, batch["case"])
                    loss = v_loss + args.image_l1_weight * image_metrics["image_l1_weighted"]
                global_batch, finite_ranks = context.sum([pet.shape[0], int(torch.isfinite(loss))])
                if finite_ranks != context.world_size:
                    raise ValueError("Non-finite training loss")
                # DDP averages rank gradients, so weight uneven local tails by sample count.
                backward_loss = loss * (pet.shape[0] * context.world_size / global_batch)
                scaler.scale(backward_loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                learning_schedule.step()
                with torch.no_grad():
                    for averaged, current in zip(ema.parameters(), model.parameters(), strict=True):
                        averaged.lerp_(current, 1 - args.ema_decay)
                step += 1
                batch_offset = index + 1
                if step == 1 or step % args.log_interval == 0 or step == target_step:
                    components = {"v_loss": v_loss, **image_metrics}
                    if image_loss is not None:
                        components["total_loss"] = loss
                    values = context.sum([float(value.detach()) * pet.shape[0] for value in components.values()] + [pet.shape[0]])
                    sample_count = values[-1]
                    logged_losses = {f"train/{key}": value / sample_count for key, value in zip(components, values[:-1], strict=True)}
                    context.on_main(lambda: logger.log({**logged_losses,
                        "train/learning_rate": optimizer.param_groups[0]["lr"], "train/grad_norm": float(grad_norm),
                        "train/batch_size": int(sample_count), "train/world_size": context.world_size}, step))
                    if context.is_main:
                        print(f"step {step}/{args.max_steps}: " + ", ".join(
                              f"{key}={value:.6f}" for key, value in logged_losses.items()))
                if step % args.validation_interval == 0 or step == target_step:
                    metrics = validate(ema, scheduler, val_loader, device, args.seed, context, image_loss, args.image_l1_weight)
                    context.on_main(lambda: logger.log({f"val/{key}": value for key, value in metrics.items()}, step))
                    if metrics["v_loss"] < best_loss:
                        best_loss = metrics["v_loss"]
                        save([args.output_dir / "best.pt"])
                if args.validation_samples and step % args.sample_interval == 0:
                    context.on_main(lambda: evaluate_generated(ema, scheduler, datasets["validation"],
                        args.output_dir / "samples" / "validation" / f"step_{step:09d}", args.seed,
                        args.validation_samples, device, args.reference_root, args.pet_checkpoint,
                        args.decode_quantize, logger, "val", step))
                if step % args.checkpoint_interval == 0:
                    save([args.output_dir / f"step_{step:09d}.pt", args.output_dir / "last.pt"])
                if step >= target_step:
                    break
            if batch_offset == len(train_loader):
                epoch, batch_offset = epoch + 1, 0
        save([args.output_dir / "last.pt"])
        if step == args.max_steps:
            best_path = args.output_dir / "best.pt"
            if not best_path.is_file():
                raise FileNotFoundError("No best.pt available for final test")
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            ema.load_state_dict(best["ema"], strict=True)
            best_step = best["global_step"]
            del best
            metrics = validate(ema, scheduler, test_loader, device, args.seed, context, image_loss, args.image_l1_weight)
            context.on_main(lambda: logger.log({f"test/{key}": value for key, value in metrics.items()} |
                       {"test/checkpoint_step": best_step}, step))
            if args.test_samples:
                context.on_main(lambda: evaluate_generated(ema, scheduler, datasets["test"],
                    args.output_dir / "samples" / "test" / f"step_{best_step:09d}", args.seed,
                    args.test_samples, device, args.reference_root, args.pet_checkpoint,
                    args.decode_quantize, logger, "test", step))
        summary = {"global_step": step, "best_validation_loss": best_loss,
                   "completed": step == args.max_steps, "wandb_run_id": run_id, "world_size": context.world_size}
        def write_summary():
            write_json(args.output_dir / "summary.json", summary)
            if logger.run is not None:
                logger.run.summary.update(summary)
        context.on_main(write_summary)
        failed = False
        return summary
    finally:
        if logger is not None:
            logger.finish(failed)


def main():
    train(parse_args())
