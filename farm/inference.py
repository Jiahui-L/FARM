"""FARM ARM construction with noisy masking and Euler integration (Sec. 3.3.3)."""

import math
import torch
from torch import nn
from .position_encoding import absolute_position_encoding_3d
from .noisy_masking import select_visible_patches
from .model import FARMNetwork


class FARM(nn.Module):
    """FARM construction pipeline: R_cond -> R_hat with flow-matching inference."""

    def __init__(
        self,
        model_config,
        initial_noise_std=2.0,
        velocity_epsilon=0.05,
        num_denoising_steps=1,
        guidance_scale=1.0,
        guidance_start=0.0,
        guidance_end=1.0,
    ):
        super().__init__()
        self.model_config = dict(model_config)
        self.sampling_config = dict(
            initial_noise_std=initial_noise_std,
            velocity_epsilon=velocity_epsilon,
            num_denoising_steps=num_denoising_steps,
            guidance_scale=guidance_scale,
            guidance_start=guidance_start,
            guidance_end=guidance_end,
        )
        if not all((math.isfinite(v) for v in self.sampling_config.values())):
            raise ValueError("Sampling settings must be finite")
        if (
            isinstance(num_denoising_steps, bool)
            or not isinstance(num_denoising_steps, int)
            or num_denoising_steps < 1
        ):
            raise ValueError("num_denoising_steps must be a positive integer")
        if (
            initial_noise_std < 0
            or not 0 < velocity_epsilon <= 1
            or (not 0 <= guidance_start <= guidance_end <= 1)
        ):
            raise ValueError(
                "Invalid initial noise, velocity epsilon, or guidance interval"
            )
        if (
            model_config.get("input_channels", 4) != 4
            or model_config.get("arm_channels", 1) != 1
        ):
            raise ValueError(
                "FARM inference uses four input channels and one ARM output"
            )
        for key, default in (
            ("arm_shape", (2, 512, 512)),
            ("patch_shape", (2, 32, 32)),
        ):
            spatial_shape = model_config.get(key, default)
            if len(spatial_shape) != 3 or any(
                (
                    isinstance(v, bool) or not isinstance(v, int) or v <= 0
                    for v in spatial_shape
                )
            ):
                raise ValueError(f"{key} must contain three positive integers")
        hidden_dim, attention_heads = (
            model_config.get("hidden_dim", 768),
            model_config.get("attention_heads", 12),
        )
        if (
            attention_heads <= 0
            or hidden_dim % attention_heads
            or hidden_dim % 2
            or hidden_dim // attention_heads % 2
            or (hidden_dim // attention_heads < 6)
        ):
            raise ValueError(
                "Embedding and head dimensions must support three even RoPE axes"
            )
        self.network = FARMNetwork(**self.model_config)
        self.patch_shape = tuple(self.network.patch_shape)
        self.initial_noise_std, self.velocity_epsilon = (
            initial_noise_std,
            velocity_epsilon,
        )
        self.num_denoising_steps, self.guidance_scale = (
            num_denoising_steps,
            guidance_scale,
        )
        self.guidance_interval = (guidance_start, guidance_end)
        self.eval()

    @classmethod
    def from_checkpoint(cls, path, model_config, device="cpu", **sampling_settings):
        """Load model tensors using the supplied architecture configuration."""
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state_dict, dict) or not all(
            isinstance(value, torch.Tensor) for value in state_dict.values()
        ):
            raise ValueError(
                "Expected a model state dictionary containing tensors only"
            )
        model = cls(model_config, **sampling_settings)
        model.load_state_dict(state_dict, strict=True)
        return model.to(device).eval()

    def partition_voxel_patches(self, voxel_grids, num_channels):
        batch_size, _, height, length, width = voxel_grids.shape
        patch_height, patch_length, patch_width = self.patch_shape
        grid_height, grid_length, grid_width = (
            height // patch_height,
            length // patch_length,
            width // patch_width,
        )
        voxel_grids = voxel_grids.reshape(
            batch_size,
            num_channels,
            grid_height,
            patch_height,
            grid_length,
            patch_length,
            grid_width,
            patch_width,
        )
        return (
            voxel_grids.permute(0, 2, 4, 6, 1, 3, 5, 7)
            .contiguous()
            .view(batch_size, grid_height * grid_length * grid_width, -1)
        )

    @torch.no_grad()
    def reconstruct_arm(
        self,
        conditioned_arm,
        mask_probability=0.75,
        visible_patch_indices=None,
        noisy_patch_indices=None,
    ):
        """Construct normalized R_hat [batch,1,H,L,W] from R_cond.

        Visible/noisy patch indices represent P_vis / P_noi. Masked state Z_t
        is advanced using the predicted velocity in Eq. (4)."""
        if self.training:
            raise ValueError("Call model.eval() before ARM construction")
        if (
            conditioned_arm.ndim != 5
            or conditioned_arm.shape[1] != 4
            or (not conditioned_arm.is_floating_point())
            or (conditioned_arm.shape[0] < 1)
        ):
            raise ValueError("Expected floating point input [batch,4,H,L,W]")
        if any(
            (
                size <= 0 or size % patch
                for size, patch in zip(conditioned_arm.shape[-3:], self.patch_shape)
            )
        ):
            raise ValueError(
                f"Input dimensions must be divisible by patch_shape={self.patch_shape}"
            )
        if not torch.isfinite(conditioned_arm).all():
            raise ValueError("Input contains non-finite values")
        if not math.isfinite(mask_probability) or not 0 <= mask_probability <= 1:
            raise ValueError("mask_probability must be in [0, 1]")
        if (visible_patch_indices is None) != (noisy_patch_indices is None):
            raise ValueError(
                "Provide both visible_patch_indices and noisy_patch_indices"
            )
        batch_size = conditioned_arm.shape[0]
        arm, environment_grids = (conditioned_arm[:, :1], conditioned_arm[:, 1:])
        arm_patches, environment_patches = (
            self.partition_voxel_patches(arm, 1),
            self.partition_voxel_patches(environment_grids, 3),
        )
        num_patches = arm_patches.shape[1]
        if visible_patch_indices is not None:
            for patch_indices in (visible_patch_indices, noisy_patch_indices):
                if (
                    patch_indices.dtype != torch.long
                    or patch_indices.device != conditioned_arm.device
                    or patch_indices.ndim != 2
                    or (patch_indices.shape[0] != batch_size)
                ):
                    raise ValueError(
                        "Patch indices must be long tensors [batch,N_p] on the input device"
                    )
            all_patch_indices = torch.cat(
                [visible_patch_indices, noisy_patch_indices], dim=1
            )
            full_patch_indices = torch.arange(
                num_patches, device=conditioned_arm.device
            ).expand(batch_size, num_patches)
            if all_patch_indices.shape != full_patch_indices.shape or not torch.equal(
                all_patch_indices.sort(dim=1).values, full_patch_indices
            ):
                raise ValueError(
                    "Patch indices must partition the entire grid without duplicates"
                )
        noisy_arm = (
            torch.randn(
                batch_size,
                num_patches,
                self.network.arm_patch_dim,
                device=conditioned_arm.device,
            )
            * self.initial_noise_std
        )
        denoising_times = torch.linspace(
            0.0, 1.0, self.num_denoising_steps + 1, device=conditioned_arm.device
        )
        if visible_patch_indices is None:
            embedded_patches = self.network.encoder_arm_embedding(conditioned_arm)
            absolute_positions = absolute_position_encoding_3d(
                self.network.hidden_dim,
                self.network.encoder_arm_embedding.patch_grid_shape,
            )
            tokens = embedded_patches + absolute_positions.float().unsqueeze(0).to(
                conditioned_arm.device
            )
            _, _, _, visible_patch_indices, noisy_patch_indices = (
                select_visible_patches(
                    tokens,
                    mask_probability,
                    self.network.encoder_arm_embedding.patch_grid_shape,
                )
            )
        noisy_indices = noisy_patch_indices.unsqueeze(-1).expand(
            -1, -1, self.network.arm_patch_dim
        )
        for current_time, next_time in zip(denoising_times[:-1], denoising_times[1:]):
            masked_arm_patches = arm_patches.clone()
            current_noisy_patches = torch.gather(noisy_arm, 1, noisy_indices)
            masked_arm_patches.scatter_(1, noisy_indices, current_noisy_patches)

            def predict(condition_free=False):
                decoder_conditioned_arm, decoder_environment_patches = (
                    conditioned_arm,
                    environment_patches,
                )
                if condition_free:
                    decoder_conditioned_arm = conditioned_arm.clone()
                    decoder_conditioned_arm[:, 1:] = -2.0
                    decoder_environment_patches = torch.full_like(
                        environment_patches, -2.0
                    )
                noisy_conditioned_patches = torch.cat(
                    [masked_arm_patches, decoder_environment_patches], dim=-1
                )
                reconstructed_patches, _ = self.network(
                    decoder_conditioned_arm,
                    current_time.expand(batch_size),
                    mask_probability=mask_probability,
                    noisy_conditioned_patches=noisy_conditioned_patches,
                    visible_patch_indices=visible_patch_indices,
                    noisy_patch_indices=noisy_patch_indices,
                )
                return reconstructed_patches

            apply_guidance = (
                abs(self.guidance_scale - 1.0) > 1e-06
                and self.guidance_interval[0]
                <= current_time.item()
                <= self.guidance_interval[1]
            )
            if apply_guidance and abs(self.guidance_scale) < 1e-06:
                reconstructed_patches = predict(condition_free=True)
            elif apply_guidance:
                conditioned_prediction = predict()
                condition_free_prediction = predict(condition_free=True)
                reconstructed_patches = (
                    condition_free_prediction
                    + self.guidance_scale
                    * (conditioned_prediction - condition_free_prediction)
                )
            else:
                reconstructed_patches = predict()
            predicted_velocity = (
                torch.gather(reconstructed_patches, 1, noisy_indices)
                - current_noisy_patches
            ) / (1 - current_time).clamp_min(self.velocity_epsilon)
            noisy_arm.scatter_(
                1,
                noisy_indices,
                current_noisy_patches + (next_time - current_time) * predicted_velocity,
            )
        reconstructed_arm_patches = arm_patches.clone()
        reconstructed_arm_patches.scatter_(
            1, noisy_indices, torch.gather(noisy_arm, 1, noisy_indices)
        )
        return (
            self.network.restore_arm_volume(reconstructed_arm_patches),
            noisy_patch_indices,
        )
