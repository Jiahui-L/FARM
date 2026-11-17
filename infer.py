"""Reconstruct a 3D aerial radio map using environmental conditions,
received signal strength (RSS) measurements, or both.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from farm import FARM
from farm.environment_conditioning import (
    decode_arm,
    load_dataset_sample,
    prepare_environment_conditioning,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to model weights (.pth)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Architecture configuration (.json) matching the model weights",
    )
    parser.add_argument(
        "--input-setting",
        required=True,
        choices=("condition-free", "condition-only", "condition-and-sample"),
        help="Use RSS measurements only, environmental conditions only, or both",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        help="Dataset ARM .npy file in its downloaded directory structure",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=0.05,
        help=(
            "For condition-free/condition-and-sample with --sample, randomly use "
            "this fraction of complete ARM patches as RSS measurements "
            "(0.05 means 5%%; default: 0.05)"
        ),
    )
    parser.add_argument(
        "--environment-grids",
        type=Path,
        help="Condition .npy [3,H,L,W]: antenna-aware FSPL, building occupancy, and base station position",
    )
    parser.add_argument(
        "--arm", type=Path, help="Encoded RSS measurements .npy [1,H,L,W] or [H,L,W]"
    )
    parser.add_argument(
        "--observed-patch-mask",
        type=Path,
        help="Binary .npy grid selecting complete RSS measurement patches (1 = observed)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Save the reconstructed ARM to this .npz file (must not already exist)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device, e.g. cpu, cuda, or cuda:0",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed for measurement selection and inference (default: 0)",
    )
    parser.add_argument(
        "--denoising-steps", type=int, help="Number of denoising steps (default: 1)"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        help="Classifier-free guidance strength (default: 1.0)",
    )
    parser.add_argument(
        "--initial-noise-std",
        type=float,
        help="Standard deviation of the initial Gaussian noise (default: 2.0)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Run CUDA inference with bfloat16 mixed precision",
    )
    args = parser.parse_args()
    if args.sample is not None and (
        args.arm is not None or args.environment_grids is not None
    ):
        parser.error("--sample cannot be combined with --arm or --environment-grids")
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    device = torch.device(args.device)
    if args.amp and device.type != "cuda":
        parser.error("--amp requires a CUDA device")
    overrides = {
        key: value
        for key, value in (
            ("num_denoising_steps", args.denoising_steps),
            ("guidance_scale", args.guidance_scale),
            ("initial_noise_std", args.initial_noise_std),
        )
        if value is not None
    }
    model_config = json.loads(args.config.read_text())
    model = FARM.from_checkpoint(
        args.checkpoint, model_config, device=device, **overrides
    )
    if args.sample is not None:
        arrays = load_dataset_sample(
            args.sample,
            args.input_setting,
            model.patch_shape,
            sampling_rate=args.sampling_rate,
            seed=args.seed,
            observed_patch_mask=(
                np.load(args.observed_patch_mask, allow_pickle=False)
                if args.observed_patch_mask is not None
                else None
            ),
        )
    else:
        arrays = {
            name: np.load(path, allow_pickle=False) if path is not None else None
            for name, path in (
                ("environment_grids", args.environment_grids),
                ("arm", args.arm),
                ("observed_patch_mask", args.observed_patch_mask),
            )
        }
    conditioned_arm, visible_patch_indices, noisy_patch_indices, visible_voxel_mask = (
        prepare_environment_conditioning(
            args.input_setting, model.patch_shape, **arrays
        )
    )
    torch.manual_seed(args.seed)
    with (
        torch.inference_mode(),
        torch.autocast(device.type, enabled=args.amp, dtype=torch.bfloat16),
    ):
        reconstructed_arm, _ = model.reconstruct_arm(
            conditioned_arm.to(device),
            visible_patch_indices=visible_patch_indices.to(device),
            noisy_patch_indices=noisy_patch_indices.to(device),
        )
        reconstructed_arm = decode_arm(reconstructed_arm).float().cpu().numpy()[0]
    if not np.isfinite(reconstructed_arm).all():
        raise ValueError("Inference produced non-finite predictions")
    metadata = dict(
        input_setting=args.input_setting,
        seed=args.seed,
        device=str(device),
        amp=args.amp,
        observed_patch_fraction=float(visible_voxel_mask.mean()),
        model_config=model.model_config,
        sampling_config=model.sampling_config,
        output_encoding="ARM pixel values in [135,255]",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.savez_compressed(
            handle,
            reconstructed_arm=reconstructed_arm,
            visible_voxel_mask=visible_voxel_mask,
            metadata=np.asarray(json.dumps(metadata)),
        )
    print(f"Saved {args.output}: reconstructed ARM {reconstructed_arm.shape}")


if __name__ == "__main__":
    main()
