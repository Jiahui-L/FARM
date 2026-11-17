"""FARM network: ARM embedding, radio encoder, and map decoder (Sec. 3.2).

Tensor storage is [batch, channels, height, length, width].
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
from .position_encoding import (
    absolute_position_encoding_3d,
    rotary_position_encoding_3d,
    apply_rotary_position_encoding,
)
from .noisy_masking import select_visible_patches


class RMSNorm(nn.Module):
    """RMS normalization used by the map decoder, Eq. (15)."""

    def __init__(self, hidden_dim, epsilon=1e-06, accumulate_float32=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.epsilon = epsilon
        self.accumulate_float32 = accumulate_float32

    def forward(self, features):
        if self.accumulate_float32:
            input_dtype = features.dtype
            features = features.to(torch.float32)
            variance = features.pow(2).mean(-1, keepdim=True)
            features = features * torch.rsqrt(variance + self.epsilon)
            return (self.weight.to(features.dtype) * features).to(input_dtype)
        variance = features.pow(2).mean(-1, keepdim=True)
        features = features * torch.rsqrt(variance + self.epsilon)
        return self.weight.to(features.dtype) * features


def apply_adaptive_modulation(features, beta, gamma):
    return features * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class TimestepEmbedding(nn.Module):
    """Diffusion timestep embedding t_dec, Sec. 3.2.5."""

    def __init__(self, hidden_dim, frequency_dim=256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.frequency_dim = frequency_dim

    @staticmethod
    def sinusoidal_timestep_encoding(time, feature_dim, max_period=10000):
        """Sinusoidal features adapted from OpenAI GLIDE's timestep_embedding.

        https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py

        MIT License
        Copyright (c) 2021 OpenAI

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        """
        half_dim = feature_dim // 2
        angular_frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=time.device)
        phase_angles = time[:, None].float() * angular_frequencies[None]
        time_encoding = torch.cat(
            [torch.cos(phase_angles), torch.sin(phase_angles)], dim=-1
        )
        if feature_dim % 2:
            time_encoding = torch.cat(
                [time_encoding, torch.zeros_like(time_encoding[:, :1])], dim=-1
            )
        return time_encoding

    def forward(self, time):
        time_encoding = self.sinusoidal_timestep_encoding(time, self.frequency_dim)
        decoder_time_embedding = self.projection(time_encoding)
        return decoder_time_embedding


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE, Eqs. (12) and (16)."""

    def __init__(
        self,
        feature_dim,
        attention_heads=8,
        qkv_bias=True,
        normalize_queries_and_keys=True,
        attention_dropout=0.0,
        projection_dropout=0.0,
    ):
        super().__init__()
        self.attention_heads = attention_heads
        head_dim = feature_dim // attention_heads
        self.query_norm = (
            RMSNorm(head_dim) if normalize_queries_and_keys else nn.Identity()
        )
        self.key_norm = (
            RMSNorm(head_dim) if normalize_queries_and_keys else nn.Identity()
        )
        self.qkv_projection = nn.Linear(feature_dim, feature_dim * 3, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_projection = nn.Linear(feature_dim, feature_dim)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(self, features, rotary_positions=None):
        batch_size, num_tokens, hidden_dim = features.shape
        qkv_features = (
            self.qkv_projection(features)
            .reshape(
                batch_size,
                num_tokens,
                3,
                self.attention_heads,
                hidden_dim // self.attention_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        queries, keys, values = (qkv_features[0], qkv_features[1], qkv_features[2])
        queries = self.query_norm(queries)
        keys = self.key_norm(keys)
        queries = apply_rotary_position_encoding(queries, rotary_positions)
        keys = apply_rotary_position_encoding(keys, rotary_positions)
        attention_scale = 1 / math.sqrt(queries.size(-1))
        dropout_probability = self.attention_dropout.p if self.training else 0.0
        attention_bias = torch.zeros(
            queries.size(0),
            1,
            queries.size(-2),
            keys.size(-2),
            dtype=torch.float32,
            device=queries.device,
        )
        attention_weights = (
            queries.float() @ keys.float().transpose(-2, -1) * attention_scale
        )
        attention_weights += attention_bias
        attention_weights = torch.softmax(attention_weights, dim=-1)
        attention_weights = torch.dropout(
            attention_weights, dropout_probability, train=self.training
        )
        features = attention_weights @ values.float()
        features = features.transpose(1, 2).reshape(batch_size, num_tokens, hidden_dim)
        features = self.output_projection(features)
        features = self.projection_dropout(features)
        return features


class SwiGLUFeedForward(nn.Module):
    """Gated feed-forward sublayer of a map-decoder block."""

    def __init__(
        self, feature_dim: int, intermediate_dim: int, dropout=0.0, bias=True
    ) -> None:
        super().__init__()
        intermediate_dim = int(intermediate_dim * 2 / 3)
        self.gate_value_projection = nn.Linear(
            feature_dim, 2 * intermediate_dim, bias=bias
        )
        self.output_projection = nn.Linear(intermediate_dim, feature_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        gate_value_features = self.gate_value_projection(features)
        gate_features, value_features = gate_value_features.chunk(2, dim=-1)
        gated_features = F.silu(gate_features) * value_features
        return self.output_projection(self.dropout(gated_features))


class FeedForwardNetwork(nn.Module):
    """Feed-forward sublayer of a radio-encoder block."""

    def __init__(
        self,
        input_dim,
        hidden_dim=None,
        output_dim=None,
        activation_layer=nn.GELU,
        dropout=0.0,
    ):
        super().__init__()
        output_dim = output_dim or input_dim
        hidden_dim = hidden_dim or input_dim
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.activation = activation_layer()
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features):
        features = self.input_projection(features)
        features = self.activation(features)
        features = self.dropout(features)
        features = self.output_projection(features)
        features = self.dropout(features)
        return features


class BottleneckARMEmbedding(nn.Module):
    """ARM patch embedding with a convolutional bottleneck."""

    def __init__(
        self,
        arm_shape=(2, 512, 512),
        patch_shape=(2, 32, 32),
        input_channels=4,
        embedding_bottleneck_dim=128,
        hidden_dim=768,
    ):
        super().__init__()
        self.arm_shape = tuple(arm_shape)
        self.patch_shape = patch_shape
        self.patch_grid_shape = (
            arm_shape[0] // patch_shape[0],
            arm_shape[1] // patch_shape[1],
            arm_shape[2] // patch_shape[2],
        )
        self.num_patches = (
            self.patch_grid_shape[0]
            * self.patch_grid_shape[1]
            * self.patch_grid_shape[2]
        )
        self.patch_feature_dim = (
            input_channels * patch_shape[0] * patch_shape[1] * patch_shape[2]
        )
        self.bottleneck_projection = nn.Conv3d(
            input_channels,
            embedding_bottleneck_dim,
            kernel_size=patch_shape,
            stride=patch_shape,
            bias=False,
        )
        self.token_projection = nn.Conv3d(
            embedding_bottleneck_dim, hidden_dim, kernel_size=1, stride=1, bias=True
        )

    def forward(self, conditioned_arm):
        batch_size, input_channels, height, length, width = conditioned_arm.shape
        conditioned_arm = self.bottleneck_projection(conditioned_arm)
        conditioned_arm = self.token_projection(conditioned_arm)
        grid_height, grid_length, grid_width = (
            conditioned_arm.shape[2],
            conditioned_arm.shape[3],
            conditioned_arm.shape[4],
        )
        self.patch_grid_shape = (grid_height, grid_length, grid_width)
        self.num_patches = grid_height * grid_length * grid_width
        conditioned_arm = conditioned_arm.permute(0, 2, 3, 4, 1).contiguous()
        conditioned_arm = conditioned_arm.view(batch_size, self.num_patches, -1)
        return conditioned_arm


class ARMEmbedding(nn.Module):
    """Encoder ARM Embedding: E-MLP for the visible patch set, Eq. (10)."""

    def __init__(
        self,
        arm_shape=(2, 512, 512),
        patch_shape=(2, 32, 32),
        input_channels=4,
        hidden_dim=768,
    ):
        super().__init__()
        self.arm_shape = arm_shape
        self.patch_shape = patch_shape
        self.patch_grid_shape = (
            arm_shape[0] // patch_shape[0],
            arm_shape[1] // patch_shape[1],
            arm_shape[2] // patch_shape[2],
        )
        self.num_patches = (
            self.patch_grid_shape[0]
            * self.patch_grid_shape[1]
            * self.patch_grid_shape[2]
        )
        self.patch_feature_dim = (
            input_channels * patch_shape[0] * patch_shape[1] * patch_shape[2]
        )
        self.projection = nn.Linear(self.patch_feature_dim, hidden_dim)

    def forward(self, conditioned_arm):
        batch_size, input_channels, height, length, width = conditioned_arm.shape
        patch_height, patch_length, patch_width = self.patch_shape
        grid_height = height // patch_height
        grid_length = length // patch_length
        grid_width = width // patch_width
        num_patches = grid_height * grid_length * grid_width
        self.patch_grid_shape = (grid_height, grid_length, grid_width)
        self.num_patches = num_patches
        conditioned_arm = conditioned_arm.view(
            batch_size,
            input_channels,
            grid_height,
            patch_height,
            grid_length,
            patch_length,
            grid_width,
            patch_width,
        )
        conditioned_arm = conditioned_arm.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        conditioned_arm = conditioned_arm.view(batch_size, num_patches, -1)
        conditioned_arm = self.projection(conditioned_arm)
        return conditioned_arm


class ARMReconstructionHead(nn.Module):
    """Project decoder features to clean ARM patches, Eq. (17)."""

    def __init__(self, hidden_dim, patch_feature_dim, arm_channels=1):
        super().__init__()
        self.output_norm = RMSNorm(hidden_dim)
        self.arm_projection = nn.Linear(hidden_dim, patch_feature_dim, bias=True)
        self.adaptive_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    def forward(self, full_features, decoder_time_embedding):
        beta, gamma = self.adaptive_modulation(decoder_time_embedding).chunk(2, dim=1)
        full_features = apply_adaptive_modulation(
            self.output_norm(full_features), beta, gamma
        )
        full_features = self.arm_projection(full_features)
        return full_features


class RadioEncoderBlock(nn.Module):
    """ViT radio-encoder block for F_vis / F_enc, Sec. 3.2.4."""

    def __init__(
        self,
        feature_dim,
        attention_heads,
        ffn_expansion_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        projection_dropout=0.0,
        residual_dropout=0.0,
    ):
        super().__init__()
        self.attention_norm = nn.LayerNorm(feature_dim)
        self.self_attention = MultiHeadSelfAttention(
            feature_dim,
            attention_heads=attention_heads,
            qkv_bias=False,
            normalize_queries_and_keys=False,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
        )
        self.ffn_norm = nn.LayerNorm(feature_dim)
        self.feed_forward = FeedForwardNetwork(
            input_dim=feature_dim,
            hidden_dim=int(feature_dim * ffn_expansion_ratio),
            dropout=dropout,
        )

    def forward(self, visible_features, rotary_positions=None):
        visible_features = visible_features + self.self_attention(
            self.attention_norm(visible_features), rotary_positions=rotary_positions
        )
        visible_features = visible_features + self.feed_forward(
            self.ffn_norm(visible_features)
        )
        return visible_features


class MapDecoderBlock(nn.Module):
    """DiT map-decoder block with AdaLN-Zero, Sec. 3.2.5."""

    def __init__(
        self,
        hidden_dim,
        attention_heads,
        ffn_expansion_ratio=4.0,
        attention_dropout=0.0,
        projection_dropout=0.0,
    ):
        super().__init__()
        self.attention_norm = RMSNorm(hidden_dim, epsilon=1e-06)
        self.self_attention = MultiHeadSelfAttention(
            hidden_dim,
            attention_heads=attention_heads,
            qkv_bias=True,
            normalize_queries_and_keys=True,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
        )
        self.ffn_norm = RMSNorm(hidden_dim, epsilon=1e-06)
        ffn_hidden_dim = int(hidden_dim * ffn_expansion_ratio)
        self.feed_forward = SwiGLUFeedForward(
            hidden_dim, ffn_hidden_dim, dropout=projection_dropout
        )
        self.adaptive_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    def forward(self, full_features, decoder_time_embedding, rotary_positions=None):
        (
            beta_attention,
            gamma_attention,
            alpha_attention,
            beta_ffn,
            gamma_ffn,
            alpha_ffn,
        ) = self.adaptive_modulation(decoder_time_embedding).chunk(6, dim=-1)
        full_features = full_features + alpha_attention.unsqueeze(
            1
        ) * self.self_attention(
            apply_adaptive_modulation(
                self.attention_norm(full_features), beta_attention, gamma_attention
            ),
            rotary_positions=rotary_positions,
        )
        full_features = full_features + alpha_ffn.unsqueeze(1) * self.feed_forward(
            apply_adaptive_modulation(self.ffn_norm(full_features), beta_ffn, gamma_ffn)
        )
        return full_features


class FARMNetwork(nn.Module):
    """ARM Embedding, Radio Encoder, and Map Decoder in the FARM network."""

    def __init__(
        self,
        arm_shape=(2, 512, 512),
        patch_shape=(2, 32, 32),
        input_channels=4,
        arm_channels=1,
        hidden_dim=768,
        radio_encoder_depth=12,
        map_decoder_depth=12,
        attention_heads=12,
        ffn_expansion_ratio=4.0,
        embedding_bottleneck_dim=128,
        attention_dropout=0.0,
        projection_dropout=0.0,
        use_embedding_bottleneck=False,
        use_global_token=True,
    ):
        super().__init__()
        self.arm_shape = arm_shape
        self.patch_shape = patch_shape
        self.arm_channels = arm_channels
        self.use_global_token = use_global_token
        self.attention_heads = attention_heads
        self.conditioned_patch_dim = (
            input_channels * patch_shape[0] * patch_shape[1] * patch_shape[2]
        )
        self.arm_patch_dim = (
            arm_channels * patch_shape[0] * patch_shape[1] * patch_shape[2]
        )
        if use_embedding_bottleneck:
            self.encoder_arm_embedding = BottleneckARMEmbedding(
                arm_shape,
                patch_shape,
                input_channels,
                embedding_bottleneck_dim,
                hidden_dim,
            )
        else:
            self.encoder_arm_embedding = ARMEmbedding(
                arm_shape, patch_shape, input_channels, hidden_dim
            )
        self.hidden_dim = hidden_dim
        if use_global_token:
            self.encoder_global_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.encoder_global_position = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        else:
            self.encoder_global_token = None
            self.encoder_global_position = None
        self.radio_encoder = nn.ModuleList(
            [
                RadioEncoderBlock(
                    hidden_dim,
                    attention_heads,
                    ffn_expansion_ratio=ffn_expansion_ratio,
                    attention_dropout=attention_dropout,
                    projection_dropout=projection_dropout,
                )
                for _ in range(radio_encoder_depth)
            ]
        )
        self.radio_encoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder_arm_embedding = nn.Linear(self.conditioned_patch_dim, hidden_dim)
        self.timestep_embedding = TimestepEmbedding(hidden_dim)
        self.map_decoder = nn.ModuleList(
            [
                MapDecoderBlock(
                    hidden_dim,
                    attention_heads,
                    ffn_expansion_ratio=ffn_expansion_ratio,
                    attention_dropout=attention_dropout,
                    projection_dropout=projection_dropout,
                )
                for _ in range(map_decoder_depth)
            ]
        )
        self.arm_reconstruction = ARMReconstructionHead(
            hidden_dim, self.arm_patch_dim, arm_channels
        )
        self.register_buffer("rotary_cosine", None, persistent=False)
        self.register_buffer("rotary_sine", None, persistent=False)
        self.initialize_parameters()

    def initialize_parameters(self):
        if hasattr(self.encoder_arm_embedding, "projection") and isinstance(
            self.encoder_arm_embedding.projection, nn.Linear
        ):
            nn.init.xavier_uniform_(self.encoder_arm_embedding.projection.weight)
        elif hasattr(self.encoder_arm_embedding, "bottleneck_projection"):
            nn.init.xavier_uniform_(
                self.encoder_arm_embedding.bottleneck_projection.weight
            )
            nn.init.xavier_uniform_(self.encoder_arm_embedding.token_projection.weight)
        nn.init.xavier_uniform_(self.decoder_arm_embedding.weight)
        if self.encoder_global_token is not None:
            nn.init.trunc_normal_(self.encoder_global_token, std=0.02)
            nn.init.trunc_normal_(self.encoder_global_position, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
        for block in self.map_decoder:
            nn.init.constant_(block.adaptive_modulation[-1].weight, 0)
            nn.init.constant_(block.adaptive_modulation[-1].bias, 0)
        nn.init.constant_(self.arm_reconstruction.adaptive_modulation[-1].weight, 0)
        nn.init.constant_(self.arm_reconstruction.adaptive_modulation[-1].bias, 0)
        nn.init.constant_(self.arm_reconstruction.arm_projection.weight, 0)
        nn.init.constant_(self.arm_reconstruction.arm_projection.bias, 0)

    def restore_arm_volume(self, arm_patches):
        batch_size, num_tokens, _ = arm_patches.shape
        patch_height, patch_length, patch_width = self.patch_shape
        grid_height, grid_length, grid_width = (
            self.encoder_arm_embedding.patch_grid_shape
        )
        arm_channels = self.arm_channels
        arm_patches = arm_patches.view(
            batch_size,
            grid_height,
            grid_length,
            grid_width,
            arm_channels,
            patch_height,
            patch_length,
            patch_width,
        )
        arm_patches = arm_patches.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        arm_patches = arm_patches.view(
            batch_size,
            arm_channels,
            grid_height * patch_height,
            grid_length * patch_length,
            grid_width * patch_width,
        )
        return arm_patches

    def get_rotary_position_encoding(self, device):
        patch_grid_shape = self.encoder_arm_embedding.patch_grid_shape
        if (
            self.rotary_cosine is not None
            and hasattr(self, "cached_patch_grid_shape")
            and (self.cached_patch_grid_shape == patch_grid_shape)
        ):
            return (self.rotary_cosine, self.rotary_sine)
        rotary_cosine, rotary_sine = rotary_position_encoding_3d(
            self.hidden_dim // self.attention_heads, patch_grid_shape, device=device
        )
        self.rotary_cosine = rotary_cosine
        self.rotary_sine = rotary_sine
        self.cached_patch_grid_shape = patch_grid_shape
        return (self.rotary_cosine, self.rotary_sine)

    def visible_rotary_position_encoding(
        self, rotary_cosine, rotary_sine, visible_patch_indices, batch_size, device
    ):
        head_dim = rotary_cosine.shape[-1]
        num_visible_patches = visible_patch_indices.shape[1]
        if num_visible_patches > 0:
            gather_indices = (
                visible_patch_indices.unsqueeze(1)
                .unsqueeze(-1)
                .expand(-1, -1, -1, head_dim)
            )
            visible_cosine = torch.gather(
                rotary_cosine.expand(batch_size, -1, -1, -1), 2, gather_indices
            )
            visible_sine = torch.gather(
                rotary_sine.expand(batch_size, -1, -1, -1), 2, gather_indices
            )
        else:
            visible_cosine = torch.empty(
                batch_size, 1, 0, head_dim, device=device, dtype=rotary_cosine.dtype
            )
            visible_sine = torch.empty(
                batch_size, 1, 0, head_dim, device=device, dtype=rotary_sine.dtype
            )
        if self.use_global_token:
            global_token_cosine = torch.ones(
                batch_size, 1, 1, head_dim, device=device, dtype=rotary_cosine.dtype
            )
            global_token_sine = torch.zeros(
                batch_size, 1, 1, head_dim, device=device, dtype=rotary_sine.dtype
            )
            visible_cosine = torch.cat([global_token_cosine, visible_cosine], dim=2)
            visible_sine = torch.cat([global_token_sine, visible_sine], dim=2)
        return (visible_cosine, visible_sine)

    def forward(
        self,
        conditioned_arm,
        time,
        mask_probability=0.75,
        noisy_conditioned_patches=None,
        visible_patch_indices=None,
        noisy_patch_indices=None,
    ):
        embedded_patches = self.encoder_arm_embedding(conditioned_arm)
        absolute_positions = absolute_position_encoding_3d(
            self.hidden_dim, self.encoder_arm_embedding.patch_grid_shape
        )
        absolute_positions = (
            absolute_positions.float().unsqueeze(0).to(conditioned_arm.device)
        )
        rotary_cosine, rotary_sine = self.get_rotary_position_encoding(
            conditioned_arm.device
        )
        visible_features = embedded_patches + absolute_positions
        if visible_patch_indices is not None and noisy_patch_indices is not None:
            batch_size = conditioned_arm.shape[0]
            visible_indices = visible_patch_indices.unsqueeze(-1).expand(
                -1, -1, visible_features.shape[-1]
            )
            visible_features = torch.gather(visible_features, 1, visible_indices)
        else:
            (
                visible_features,
                noisy_patch_mask,
                restore_indices,
                visible_patch_indices,
                noisy_patch_indices,
            ) = select_visible_patches(
                visible_features,
                mask_probability,
                self.encoder_arm_embedding.patch_grid_shape,
            )
        batch_size = conditioned_arm.shape[0]
        if self.use_global_token:
            global_tokens = (
                self.encoder_global_token.expand(batch_size, -1, -1)
                + self.encoder_global_position
            )
            visible_features = torch.cat([global_tokens, visible_features], dim=1)
        visible_cosine, visible_sine = self.visible_rotary_position_encoding(
            rotary_cosine,
            rotary_sine,
            visible_patch_indices,
            batch_size,
            conditioned_arm.device,
        )
        encoder_rotary_positions = (visible_cosine, visible_sine)
        for block in self.radio_encoder:
            visible_features = block(
                visible_features, rotary_positions=encoder_rotary_positions
            )
        visible_features = self.radio_encoder_norm(visible_features)
        if self.use_global_token:
            visible_features = visible_features[:, 1:, :]
        if noisy_conditioned_patches is None:
            raise ValueError(
                "noisy_conditioned_patches is required; use FARM.reconstruct_arm() for inference"
            )
        all_noisy_features = self.decoder_arm_embedding(noisy_conditioned_patches)
        noisy_indices = noisy_patch_indices.unsqueeze(-1).expand(
            -1, -1, self.hidden_dim
        )
        noisy_features = torch.gather(all_noisy_features, 1, noisy_indices)
        noisy_features = noisy_features.to(visible_features.dtype)
        batch_size, num_visible_patches, hidden_dim = visible_features.shape
        num_noisy_patches = noisy_features.shape[1]
        num_patches = num_visible_patches + num_noisy_patches
        full_features = torch.zeros(
            batch_size,
            num_patches,
            hidden_dim,
            device=conditioned_arm.device,
            dtype=visible_features.dtype,
        )
        visible_indices = visible_patch_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        full_features.scatter_(1, visible_indices, visible_features)
        noisy_indices = noisy_patch_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
        full_features.scatter_(1, noisy_indices, noisy_features)
        full_features = full_features + absolute_positions.expand(
            batch_size, -1, -1
        ).to(full_features.dtype)
        decoder_time_embedding = self.timestep_embedding(time)
        decoder_rotary_positions = (rotary_cosine, rotary_sine)
        for block in self.map_decoder:
            full_features = block(
                full_features,
                decoder_time_embedding,
                rotary_positions=decoder_rotary_positions,
            )
        reconstructed_patches = self.arm_reconstruction(
            full_features, decoder_time_embedding
        )
        return (reconstructed_patches, noisy_patch_indices)
