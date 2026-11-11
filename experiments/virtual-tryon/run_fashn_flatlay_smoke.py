#!/usr/bin/env python3
"""Run one local, consented FASHN flat-lay try-on and write a metrics JSON."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

from fashn_vton import TryOnPipeline


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--person-image", type=Path, required=True)
    parser.add_argument("--garment-image", type=Path, required=True)
    parser.add_argument("--category", choices=("tops", "bottoms", "one-pieces"), required=True)
    parser.add_argument("--output", type=Path, default=Path("tryon-output.png"))
    parser.add_argument("--metrics", type=Path, default=Path("tryon-metrics.json"))
    parser.add_argument("--steps", type=int, choices=(20, 30, 50), default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    if not args.person_image.is_file() or not args.garment_image.is_file():
        parser.error("Both image paths must point to existing local files.")
    if not args.weights_dir.is_dir():
        parser.error("--weights-dir must point to an existing weights directory.")
    if args.require_cuda and not torch.cuda.is_available():
        parser.error("CUDA was required but is not available.")

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    person = Image.open(args.person_image).convert("RGB")
    garment = Image.open(args.garment_image).convert("RGB")

    # Importing the FASHN package itself must not eagerly import the optional parser.
    parser_loaded_before = "fashn_human_parser" in sys.modules
    started = time.perf_counter()
    pipeline = TryOnPipeline(weights_dir=str(args.weights_dir))
    sync_cuda()
    initialized = time.perf_counter()

    init_peak_allocated_mb = None
    init_peak_reserved_mb = None
    if cuda_available:
        init_peak_allocated_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
        init_peak_reserved_mb = round(torch.cuda.max_memory_reserved() / (1024 * 1024), 1)
        torch.cuda.reset_peak_memory_stats()

    result = pipeline(
        person_image=person,
        garment_image=garment,
        category=args.category,
        garment_photo_type="flat-lay",
        segmentation_free=True,
        num_samples=1,
        num_timesteps=args.steps,
        seed=args.seed,
    )
    sync_cuda()
    finished = time.perf_counter()

    parser_loaded_after = "fashn_human_parser" in sys.modules
    if parser_loaded_before or parser_loaded_after:
        raise RuntimeError(
            "Human Parser was imported during the parser-free smoke test. "
            "Do not treat this run as a parser-free result."
        )
    if not getattr(result, "images", None):
        raise RuntimeError("FASHN returned no output images.")
    if len(result.images) != 1:
        raise RuntimeError(f"Expected exactly one output image, got {len(result.images)}.")
    output_image = result.images[0]
    if not isinstance(output_image, Image.Image) or output_image.width < 1 or output_image.height < 1:
        raise RuntimeError("FASHN returned an invalid output image.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(args.output)

    metrics = {
        "mode": "parser-free-flat-lay-segmentation-free",
        "category": args.category,
        "device": device_name,
        "cuda_available": cuda_available,
        "torch_version": torch.__version__,
        "steps": args.steps,
        "seed": args.seed,
        "person_image_size": list(person.size),
        "garment_image_size": list(garment.size),
        "output_image_size": list(output_image.size),
        "pipeline_init_seconds": round(initialized - started, 3),
        "inference_seconds": round(finished - initialized, 3),
        "total_seconds": round(finished - started, 3),
        "human_parser_loaded": parser_loaded_after,
        "output_path": str(args.output),
    }
    if cuda_available:
        metrics["init_peak_allocated_mb"] = init_peak_allocated_mb
        metrics["init_peak_reserved_mb"] = init_peak_reserved_mb
        metrics["inference_peak_allocated_mb"] = round(
            torch.cuda.max_memory_allocated() / (1024 * 1024), 1
        )
        metrics["inference_peak_reserved_mb"] = round(
            torch.cuda.max_memory_reserved() / (1024 * 1024), 1
        )

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved image: {args.output}")
    print(f"Saved metrics: {args.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
