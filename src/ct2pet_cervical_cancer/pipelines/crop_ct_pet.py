"""Prepare aligned, mask-cropped CT/PET volumes for paired latent encoding."""
from __future__ import annotations
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import torch
from ..datasets.cropping import process_case_safe

from ..utils.cli import parse_config_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nii-root", type=Path, default=None)
    parser.add_argument("--seg-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--min-depth",
        type=int,
        default=80,
        help="Minimum cropped axial depth after alignment; default: 80.",
    )
    parser.add_argument(
        "--depth-multiple",
        type=int,
        default=8,
        help="Round the crop depth up to this multiple for the 8X PVAE; default: 8.",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=0,
        help="Extra aligned slices added on both sides of the non-empty mask range.",
    )
    parser.add_argument(
        "--case",
        dest="case_name",
        default=None,
        help="Process only one case directory name; default: process all cases.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output case directory.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed case instead of continuing.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of cases processed concurrently; default: up to 4.",
    )
    return parse_config_args(parser, "crop_ct_pet.json")


def main() -> None:
    args = parse_args()
    if not args.nii_root.is_dir():
        raise FileNotFoundError(args.nii_root)
    if not args.seg_root.is_dir():
        raise FileNotFoundError(args.seg_root)
    if args.workers < 1:
        raise ValueError(f"workers must be positive, got {args.workers}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.case_name:
        case_dirs = [args.nii_root / args.case_name]
    else:
        case_dirs = sorted(path for path in args.nii_root.iterdir() if path.is_dir())

    # Each resampling task is already CPU-intensive. One intra-op thread per
    # worker gives better throughput than N workers each spawning all CPU cores.
    if args.workers > 1:
        torch.set_num_threads(1)

    results: list[dict[str, Any]] = []
    if args.workers == 1:
        for case_dir in case_dirs:
            result = process_case_safe(case_dir, args)
            results.append(result)
            print(result)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_case_safe, case_dir, args): case_dir
                for case_dir in case_dirs
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(result)
                if args.fail_fast and result["status"] == "failed":
                    raise RuntimeError(result["reason"])

    results.sort(key=lambda item: item["case"])

    summary_path = args.output_root / "batch_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    counts = {status: sum(item["status"] == status for item in results) for status in {item["status"] for item in results}}
    print(f"Summary: {counts}; saved to {summary_path}")
