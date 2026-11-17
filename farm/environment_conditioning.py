"""Environment Conditioning (Sec. 3.2.2) and ARM value normalization.

Channels are [R, V_fspl, B, V_pos], with spatial axes [H,L,W].
"""

import csv
import json
from pathlib import Path
import re

import numpy as np
import torch


DATASET_HEIGHT_LAYERS = {
    "D1": 24,
    "D2": 26,
    "D3": 22,
    "D4": 22,
    "D5": 24,
    "D6": 26,
    "D7": 26,
    "D8": 20,
    "D9": 24,
    "D10": 26,
    "P1": 30,
    "F1": 24,
    "A1": 24,
}


def build_environment_grids(
    building_occupancy,
    transmitter_position,
    receiver_heights_m,
    frequency_hz,
    yaw_degrees,
    antenna_pattern,
):
    """Build [V_fspl, B, V_pos] on the dataset's 1000/512 m horizontal grid.

    Transmitter position is (row, column, height_m). Antenna gain is indexed
    by polar angle 0..180 and azimuth 0..360, in degrees and dB.
    """
    building_occupancy = np.asarray(building_occupancy, dtype=bool)
    heights = np.asarray(receiver_heights_m, dtype=np.float64)
    antenna_pattern = np.asarray(antenna_pattern)
    if building_occupancy.ndim != 3 or heights.shape != (building_occupancy.shape[0],):
        raise ValueError("Occupancy must be [H,L,W] with one receiver height per layer")
    if antenna_pattern.shape != (181, 361) or not np.isfinite(antenna_pattern).all():
        raise ValueError("Antenna pattern must be a finite [181,361] gain array")
    tx_row, tx_column, tx_height = transmitter_position
    num_layers, length, width = building_occupancy.shape
    if (
        min(building_occupancy.shape) <= 0
        or not 0 <= tx_row < length
        or not 0 <= tx_column < width
        or not np.isfinite(
            [tx_row, tx_column, tx_height, frequency_hz, yaw_degrees]
        ).all()
        or frequency_hz <= 0
        or not np.isfinite(heights).all()
    ):
        raise ValueError("Invalid transmitter, frequency, or receiver grid")
    columns, rows = np.meshgrid(np.arange(width), np.arange(length))
    dx = (columns / 512 * 1000 - 500) - (tx_column / 512 * 1000 - 500)
    dy = (500 - rows / 512 * 1000) - (500 - tx_row / 512 * 1000)
    azimuth = np.mod(np.arctan2(dy, dx) - np.deg2rad(yaw_degrees - 360), 2 * np.pi)
    azimuth_indices = np.clip((azimuth * 360 / (2 * np.pi)).astype(np.int32), 0, 360)
    grids = np.zeros((3, num_layers, length, width), dtype=np.uint8)
    for index, height in enumerate(heights):
        dz = height - tx_height
        distance = np.maximum(np.sqrt(dx**2 + dy**2 + dz**2), 1.0)
        polar_angle = np.arccos(np.clip(dz / distance, -1, 1))
        polar_indices = np.clip((polar_angle * 180 / np.pi).astype(np.int32), 0, 180)
        gain = antenna_pattern[polar_indices, azimuth_indices]
        path_loss = 20 * np.log10(distance) + 20 * np.log10(frequency_hz) - 147.55
        antenna_aware_fspl = np.where(gain > -200, gain - path_loss, -250)
        grids[0, index] = np.clip(antenna_aware_fspl + 255, 0, 255).astype(np.uint8)
    grids[1] = building_occupancy.astype(np.uint8) * 255
    grids[2, :, int(tx_row), int(tx_column)] = 255
    return grids


def load_dataset_sample(
    sample_path,
    input_setting,
    patch_shape,
    sampling_rate=0.05,
    seed=0,
    observed_patch_mask=None,
):
    """Read a released ARM sample and construct its inference inputs in memory."""
    sample_path = Path(sample_path).resolve()
    pattern_directory = sample_path.parent
    frequency_directory = pattern_directory.parent
    scene_directory = frequency_directory.parent
    dataset_directory = scene_directory.parent
    data_root = dataset_directory.parent
    sample_match = re.fullmatch(r"tx(\d+)_yaw(\d+)", sample_path.stem)
    frequency_match = re.fullmatch(r"freq(\d+)", frequency_directory.name)
    if sample_match is None or frequency_match is None:
        raise ValueError(
            "Expected <dataset>/<scene>/freq*/<pattern>/tx<ID>_yaw<ANGLE>.npy"
        )
    raw = np.load(sample_path, mmap_mode="r", allow_pickle=False)
    if (
        raw.ndim != 4
        or raw.shape[0] != 1
        or raw.shape[-2:] != (512, 512)
        or raw.dtype != np.uint8
    ):
        raise ValueError("Dataset ARM must be uint8 [1,H,512,512]")
    heights = json.loads((data_root / "meta_ranges.json").read_text())["height"]
    num_layers = DATASET_HEIGHT_LAYERS.get(dataset_directory.name, len(heights))
    if len(heights) != raw.shape[1] or num_layers > len(heights):
        raise ValueError("ARM height layers do not match dataset metadata")
    arm = raw[:, :num_layers]
    environment_grids = None
    transmitter_position = None
    if input_setting != "condition-free" or dataset_directory.name == "P1":
        transmitter_id = int(sample_match[1])
        with (scene_directory / "tx_positions_512.csv").open(newline="") as handle:
            position = next(
                (
                    row
                    for row in csv.DictReader(handle)
                    if int(row["tx_id"]) == transmitter_id
                ),
                None,
            )
        if position is None:
            raise ValueError(f"Transmitter {transmitter_id} not found in scene CSV")
        transmitter_position = tuple(
            float(position[key]) for key in ("row", "col", "height")
        )
    if input_setting != "condition-free":
        pattern_name = pattern_directory.name
        pattern = (
            np.zeros((181, 361), dtype=np.float64)
            if pattern_name == "iso"
            else np.load(
                Path(__file__).with_name("antenna_patterns") / f"{pattern_name}.npy",
                allow_pickle=False,
            )
        )
        environment_grids = build_environment_grids(
            arm[0] == 0,
            transmitter_position,
            heights[:num_layers],
            int(frequency_match[1]) * 1e8,
            int(sample_match[2]),
            pattern,
        )
    if dataset_directory.name == "P1":
        row, column, _ = transmitter_position
        top = min(max(int(row) - 128, 0), 256)
        left = min(max(int(column) - 128, 0), 256)
        arm = arm[:, :, top : top + 256, left : left + 256]
        if environment_grids is not None:
            environment_grids = environment_grids[
                :, :, top : top + 256, left : left + 256
            ]
    if input_setting == "condition-only":
        arm = None
    elif observed_patch_mask is None:
        if not np.isfinite(sampling_rate) or not 0 < sampling_rate <= 1:
            raise ValueError("sampling_rate must be in (0,1]")
        if any(size % patch for size, patch in zip(arm.shape[-3:], patch_shape)):
            raise ValueError(
                "Dataset sample dimensions must be divisible by patch_shape"
            )
        patch_grid = tuple(
            size // patch for size, patch in zip(arm.shape[-3:], patch_shape)
        )
        observed_patch_mask = np.zeros(patch_grid, dtype=bool)
        num_observed = max(1, round(observed_patch_mask.size * sampling_rate))
        indices = np.random.default_rng(seed).choice(
            observed_patch_mask.size, num_observed, replace=False
        )
        observed_patch_mask.reshape(-1)[indices] = True
    return dict(
        environment_grids=environment_grids,
        arm=arm,
        observed_patch_mask=observed_patch_mask,
    )


def normalize_conditioned_arm(conditioned_arm):
    """Normalize R_cond [batch,4,H,L,W], preserving the null-condition grid M."""
    conditioned_arm = conditioned_arm.clone().float()
    missing_conditions = conditioned_arm[:, 1:] == -2.0
    conditioned_arm[:, :1][conditioned_arm[:, :1] == 0] = 135.0
    conditioned_arm[:, :1] = (conditioned_arm[:, :1] - 135.0) / 60.0 - 1.0
    conditioned_arm[:, 1:2] = (conditioned_arm[:, 1:2] - 5.0) / 125.0 - 1.0
    conditioned_arm[:, 2:] = conditioned_arm[:, 2:] / 255.0 * 2.0 - 1.0
    conditioned_arm[:, 1:][missing_conditions] = -2.0
    return conditioned_arm


def decode_arm(reconstructed_arm):
    """Decode normalized R_hat to the checkpoint's encoded [135,255] ARM range."""
    return ((reconstructed_arm + 1.0) * 60.0 + 135.0).clamp(135.0, 255.0)


def prepare_environment_conditioning(
    input_setting,
    patch_shape,
    environment_grids=None,
    arm=None,
    observed_patch_mask=None,
):
    """Assemble R_cond and the P_vis / P_noi patch sets for one ARM.

    ARM is [1,H,L,W] or [H,L,W], conditions [3,H,L,W], and the
    observed patch mask [H/h_p,L/l_p,W/w_p]. Hidden ARM entries are discarded.
    """
    if input_setting not in (
        "condition-free",
        "condition-only",
        "condition-and-sample",
    ):
        raise ValueError(f"Unknown input setting: {input_setting}")
    if input_setting == "condition-free" and environment_grids is not None:
        raise ValueError(
            "condition-free takes ARM measurements and an observed patch mask only"
        )
    if input_setting != "condition-free" and environment_grids is None:
        raise ValueError(f"{input_setting} requires environment grids")
    if input_setting == "condition-only":
        if arm is not None or observed_patch_mask is not None:
            raise ValueError("condition-only takes environment grids only")
    elif arm is None or observed_patch_mask is None:
        raise ValueError(
            f"{input_setting} requires ARM measurements and an observed patch mask"
        )
    if environment_grids is not None:
        environment_grids = np.asarray(environment_grids, dtype=np.float32)
        if (
            environment_grids.ndim != 4
            or environment_grids.shape[0] != 3
            or (not np.isfinite(environment_grids).all())
        ):
            raise ValueError("Environment grids must be finite [3,H,L,W] data")
        arm_shape = environment_grids.shape[-3:]
    if arm is not None:
        arm = np.asarray(arm, dtype=np.float32)
        if arm.ndim == 4 and arm.shape[0] == 1:
            arm = arm[0]
        if arm.ndim != 3:
            raise ValueError("ARM must have shape [H,L,W] or [1,H,L,W]")
        if environment_grids is not None and arm.shape != arm_shape:
            raise ValueError("ARM and environment grids must share spatial dimensions")
        arm_shape = arm.shape
    if len(patch_shape) != 3 or any(
        (
            patch_extent <= 0 or extent <= 0 or extent % patch_extent
            for extent, patch_extent in zip(arm_shape, patch_shape)
        )
    ):
        raise ValueError(
            f"Volume shape {arm_shape} must be divisible by patch_shape={patch_shape}"
        )
    patch_grid_shape = tuple(
        (extent // patch_extent for extent, patch_extent in zip(arm_shape, patch_shape))
    )
    if input_setting == "condition-only":
        visible_patch_mask = np.zeros(patch_grid_shape, dtype=np.bool_)
    else:
        visible_patch_mask = np.asarray(observed_patch_mask)
        if (
            visible_patch_mask.shape != patch_grid_shape
            or not np.isin(visible_patch_mask, [0, 1]).all()
        ):
            raise ValueError(
                f"Observed patch mask must be binary with grid shape {patch_grid_shape}"
            )
        visible_patch_mask = visible_patch_mask.astype(np.bool_)
        if not visible_patch_mask.any():
            raise ValueError(
                "Sample-based inference requires at least one observed ARM patch"
            )
    visible_voxel_mask = visible_patch_mask
    for axis, patch in enumerate(patch_shape):
        visible_voxel_mask = np.repeat(visible_voxel_mask, patch, axis=axis)
    if arm is None:
        arm = np.full(arm_shape, 135.0, dtype=np.float32)
    arm = np.where(visible_voxel_mask, arm, 135.0)
    if not np.isfinite(arm).all():
        raise ValueError("Observed ARM entries must be finite")
    if environment_grids is None:
        environment_grids = np.full((3, *arm_shape), -2.0, dtype=np.float32)
    conditioned_arm_raw = np.concatenate([arm[None], environment_grids], axis=0)
    visible_patch_indices = (
        torch.from_numpy(np.flatnonzero(visible_patch_mask.reshape(-1)))
        .long()
        .unsqueeze(0)
    )
    noisy_patch_indices = (
        torch.from_numpy(np.flatnonzero(~visible_patch_mask.reshape(-1)))
        .long()
        .unsqueeze(0)
    )
    return (
        normalize_conditioned_arm(torch.from_numpy(conditioned_arm_raw).unsqueeze(0)),
        visible_patch_indices,
        noisy_patch_indices,
        visible_voxel_mask,
    )
