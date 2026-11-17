"""Visible/noisy patch partition for Noisy Masking, Sec. 3.2.1."""

import math

import torch


def select_visible_patches(patch_features, mask_probability, patch_grid_shape):
    """Partition P_vis / P_noi; the Euler sampler supplies noise to P_noi."""
    batch_size, num_patches, hidden_dim = patch_features.shape
    if num_patches != math.prod(patch_grid_shape):
        raise ValueError("Patch count does not match the spatial grid")
    if not math.isfinite(mask_probability) or not 0 <= mask_probability <= 1:
        raise ValueError("mask_probability must be in [0,1]")
    num_visible_patches = round(num_patches * (1 - mask_probability))
    shuffled_indices = torch.argsort(
        torch.rand(batch_size, num_patches, device=patch_features.device), dim=1
    )
    restore_indices = torch.argsort(shuffled_indices, dim=1)
    visible_patch_indices = shuffled_indices[:, :num_visible_patches]
    noisy_patch_indices = shuffled_indices[:, num_visible_patches:]
    visible_features = torch.gather(
        patch_features, 1, visible_patch_indices.unsqueeze(-1).repeat(1, 1, hidden_dim)
    )
    noisy_patch_mask = torch.ones(batch_size, num_patches, device=patch_features.device)
    noisy_patch_mask.scatter_(1, visible_patch_indices, 0)
    return (
        visible_features,
        noisy_patch_mask,
        restore_indices,
        visible_patch_indices,
        noisy_patch_indices,
    )
