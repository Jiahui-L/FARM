"""Hybrid 3D positional encoding (H-PE), Sec. 3.2.4.

Absolute SinCos positions and relative RoPE share the stored (H,L,W) patch
order. The axis decomposition follows WiFo: https://github.com/PKU-PCNI/WiFo.
"""

import torch


def axis_absolute_encoding(axis_dim, coordinates):
    assert axis_dim % 2 == 0
    frequencies = torch.arange(
        axis_dim // 2, dtype=torch.float32, device=coordinates.device
    )
    frequencies = 1.0 / (10000 ** (frequencies / (axis_dim // 2)))
    phase_angles = coordinates.float().unsqueeze(1) * frequencies.unsqueeze(0)
    return torch.cat([torch.sin(phase_angles), torch.cos(phase_angles)], dim=1)


def absolute_position_encoding_3d(hidden_dim, patch_grid_shape, device=None):
    """Absolute component of H-PE: P_enc / P_dec, shape [N_p,hidden_dim]."""
    grid_height, grid_length, grid_width = patch_grid_shape
    if device is None:
        device = "cpu"
    axis_dim = hidden_dim // 3
    height_dim = axis_dim - axis_dim % 2
    length_dim = axis_dim - axis_dim % 2
    width_dim = hidden_dim - height_dim - length_dim
    assert min(height_dim, length_dim, width_dim) > 0
    assert height_dim % 2 == length_dim % 2 == width_dim % 2 == 0
    height_positions = torch.arange(grid_height, dtype=torch.float32, device=device)
    length_positions = torch.arange(grid_length, dtype=torch.float32, device=device)
    width_positions = torch.arange(grid_width, dtype=torch.float32, device=device)
    height_encoding = axis_absolute_encoding(height_dim, height_positions)
    length_encoding = axis_absolute_encoding(length_dim, length_positions)
    width_encoding = axis_absolute_encoding(width_dim, width_positions)
    height_encoding = height_encoding.view(grid_height, 1, 1, height_dim).expand(
        grid_height, grid_length, grid_width, height_dim
    )
    length_encoding = length_encoding.view(1, grid_length, 1, length_dim).expand(
        grid_height, grid_length, grid_width, length_dim
    )
    width_encoding = width_encoding.view(1, 1, grid_width, width_dim).expand(
        grid_height, grid_length, grid_width, width_dim
    )
    absolute_positions = torch.cat(
        [height_encoding, length_encoding, width_encoding], dim=-1
    )
    return absolute_positions.reshape(
        grid_height * grid_length * grid_width, hidden_dim
    )


def axis_rotary_frequencies(
    axis_dim, num_positions, device=None, frequency_base=10000.0
):
    assert axis_dim % 2 == 0
    frequencies = 1.0 / (
        frequency_base
        ** (torch.arange(0, axis_dim, 2, device=device).float() / axis_dim)
    )
    coordinates = torch.arange(num_positions, device=device).float()
    return torch.outer(coordinates, frequencies)


def rotary_position_encoding_3d(
    head_dim, patch_grid_shape, device=None, frequency_base=10000.0
):
    """Relative component of H-PE: cosine/sine [1,1,N_p,head_dim]."""
    grid_height, grid_length, grid_width = patch_grid_shape
    assert head_dim % 2 == 0
    axis_dim = head_dim // 3
    height_dim = axis_dim - axis_dim % 2
    length_dim = axis_dim - axis_dim % 2
    width_dim = head_dim - height_dim - length_dim
    assert min(height_dim, length_dim, width_dim) > 0
    assert height_dim % 2 == length_dim % 2 == width_dim % 2 == 0
    height_frequencies = axis_rotary_frequencies(
        height_dim, grid_height, device, frequency_base
    )
    length_frequencies = axis_rotary_frequencies(
        length_dim, grid_length, device, frequency_base
    )
    width_frequencies = axis_rotary_frequencies(
        width_dim, grid_width, device, frequency_base
    )
    height_frequencies = height_frequencies.view(grid_height, 1, 1, -1).expand(
        grid_height, grid_length, grid_width, -1
    )
    length_frequencies = length_frequencies.view(1, grid_length, 1, -1).expand(
        grid_height, grid_length, grid_width, -1
    )
    width_frequencies = width_frequencies.view(1, 1, grid_width, -1).expand(
        grid_height, grid_length, grid_width, -1
    )
    frequencies = torch.cat(
        [height_frequencies, length_frequencies, width_frequencies], dim=-1
    )
    frequencies = frequencies.reshape(-1, head_dim // 2)
    phase_angles = torch.cat((frequencies, frequencies), dim=-1)
    return phase_angles.cos().view(1, 1, -1, head_dim), phase_angles.sin().view(
        1, 1, -1, head_dim
    )


def rotate_half_features(features):
    first_half = features[..., : features.shape[-1] // 2]
    second_half = features[..., features.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rotary_position_encoding(features, rotary_positions):
    if rotary_positions is None:
        return features
    if callable(rotary_positions):
        return rotary_positions(features)
    if isinstance(rotary_positions, (tuple, list)) and len(rotary_positions) == 2:
        rotary_cosine, rotary_sine = rotary_positions
        return features * rotary_cosine + rotate_half_features(features) * rotary_sine
    raise TypeError("rotary_positions must be None, callable, or a cosine/sine pair")
