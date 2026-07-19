from __future__ import annotations

import torch

from .heads import MultiTaskHeads


class DeepSetsOperator(torch.nn.Module):
    """Permutation-invariant DeepSets baseline (Zaheer et al., NeurIPS 2017)."""

    def __init__(self, support_dim: int, query_dim: int, hidden: int, counts: tuple[int, int, int]):
        super().__init__()
        self.phi = torch.nn.Sequential(torch.nn.Linear(support_dim, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, hidden))
        self.query = torch.nn.Sequential(torch.nn.Linear(query_dim, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, hidden))
        self.fusion = torch.nn.Sequential(torch.nn.Linear(hidden * 2, hidden), torch.nn.GELU())
        self.heads = MultiTaskHeads(hidden, counts[0], counts[1], counts[2])

    def forward(self, support: torch.Tensor, query: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.phi(support).mean(dim=1, keepdim=True).expand(-1, query.shape[1], -1)
        return self.heads(self.fusion(torch.cat([self.query(query), context], dim=-1)))


class SetTransformerOperator(torch.nn.Module):
    """Set Transformer baseline (Lee et al., ICML 2019)."""

    def __init__(self, support_dim: int, query_dim: int, hidden: int, counts: tuple[int, int, int]):
        super().__init__()
        self.support = torch.nn.Linear(support_dim, hidden)
        layer = torch.nn.TransformerEncoderLayer(hidden, 4, hidden * 4, batch_first=True, activation="gelu")
        self.encoder = torch.nn.TransformerEncoder(layer, 2)
        self.query = torch.nn.Linear(query_dim, hidden)
        self.cross = torch.nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.fusion = torch.nn.Sequential(torch.nn.Linear(hidden * 2, hidden), torch.nn.GELU())
        self.heads = MultiTaskHeads(hidden, counts[0], counts[1], counts[2])

    def forward(self, support: torch.Tensor, query: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(self.support(support))
        query_feature = self.query(query)
        local, _ = self.cross(query_feature, encoded, encoded, need_weights=False)
        return self.heads(self.fusion(torch.cat([query_feature, local], dim=-1)))


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.GroupNorm(4, out_channels),
            torch.nn.GELU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
            torch.nn.GroupNorm(4, out_channels),
            torch.nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RadioUNet(torch.nn.Module):
    """RadioUNet-style encoder-decoder baseline (Levie et al., TWC 2021)."""

    def __init__(self, in_channels: int, hidden: int, output_channels: int):
        super().__init__()
        width = max(16, hidden // 4)
        self.enc1 = ConvBlock(in_channels, width)
        self.enc2 = ConvBlock(width, width * 2)
        self.bottom = ConvBlock(width * 2, width * 4)
        self.dec2 = ConvBlock(width * 6, width * 2)
        self.dec1 = ConvBlock(width * 3, width)
        self.out = torch.nn.Conv2d(width, output_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(torch.nn.functional.max_pool2d(e1, 2))
        bottom = self.bottom(torch.nn.functional.max_pool2d(e2, 2))
        up2 = torch.nn.functional.interpolate(bottom, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))
        up1 = torch.nn.functional.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.dec1(torch.cat([up1, e1], dim=1)))


class StormRMEOperator(torch.nn.Module):
    """Task-head adaptation of the IEEE ICC 2025 STORM estimator.

    The official implementation constructs translation- and rotation-invariant features for
    every target location before attention. This adaptation preserves that defining operation
    and replaces its scalar RSS output with the benchmark's shared task heads.
    """

    def __init__(
        self,
        support_dim: int,
        query_dim: int,
        hidden: int,
        counts: tuple[int, int, int],
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        far_count, angle_count, range_count = counts
        task_payload = 1 + 3 + far_count + angle_count + range_count + 5
        self.rss_index = support_dim - task_payload
        if self.rss_index < 3:
            raise ValueError("STORM adaptation requires coordinates followed by support payload")
        pair_dim = support_dim - 3 + 9
        self.pair_encoder = torch.nn.Sequential(
            torch.nn.Linear(pair_dim, hidden),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
        )
        self.query_encoder = torch.nn.Sequential(
            torch.nn.Linear(query_dim - 3, hidden),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden),
        )
        self.cross_attention = torch.nn.MultiheadAttention(
            hidden,
            attention_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(hidden * 2, hidden),
            torch.nn.GELU(),
        )
        self.heads = MultiTaskHeads(hidden, far_count, angle_count, range_count)

    def forward(self, support: torch.Tensor, query: torch.Tensor) -> dict[str, torch.Tensor]:
        support_xyz = support[..., :3]
        query_xyz = query[..., :3]
        relative = support_xyz[:, None, :, :] - query_xyz[:, :, None, :]
        rss = support[..., self.rss_index]
        spatial_weight = torch.softmax(rss, dim=1)
        direction = torch.sum(
            relative * spatial_weight[:, None, :, None],
            dim=2,
        )
        rotation = torch.atan2(direction[..., 1], direction[..., 0])
        cosine = torch.cos(rotation)[:, :, None]
        sine = torch.sin(rotation)[:, :, None]
        rotated_x = relative[..., 0] * cosine + relative[..., 1] * sine
        rotated_y = -relative[..., 0] * sine + relative[..., 1] * cosine
        point_angle = torch.atan2(rotated_y, rotated_x)
        distance = torch.linalg.vector_norm(relative, dim=-1)
        inverse_distance = 1.0 / (distance + 0.1)
        invariant = torch.stack(
            [
                rotated_x,
                rotated_y,
                relative[..., 2],
                point_angle,
                torch.cos(point_angle),
                torch.sin(point_angle),
                distance,
                inverse_distance,
                inverse_distance.square(),
            ],
            dim=-1,
        )
        payload = support[..., 3:][:, None, :, :].expand(-1, query.shape[1], -1, -1)
        pair = self.pair_encoder(torch.cat([payload, invariant], dim=-1))
        batch, query_count, support_count, hidden = pair.shape
        query_feature = self.query_encoder(query[..., 3:])
        attended, _ = self.cross_attention(
            query_feature.reshape(batch * query_count, 1, hidden),
            pair.reshape(batch * query_count, support_count, hidden),
            pair.reshape(batch * query_count, support_count, hidden),
            need_weights=False,
        )
        attended = attended.reshape(batch, query_count, hidden)
        return self.heads(self.fusion(torch.cat([query_feature, attended], dim=-1)))


class SpectralConv2d(torch.nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / width
        self.weight = torch.nn.Parameter(scale * torch.randn(width, width, modes, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft2(x)
        output = torch.zeros_like(spectrum)
        height_modes = min(self.modes, spectrum.shape[-2])
        width_modes = min(self.modes, spectrum.shape[-1])
        output[:, :, :height_modes, :width_modes] = torch.einsum(
            "bixy,ioxy->boxy",
            spectrum[:, :, :height_modes, :width_modes],
            self.weight[:, :, :height_modes, :width_modes],
        )
        return torch.fft.irfft2(output, s=x.shape[-2:])


class FNOOperator(torch.nn.Module):
    """Fourier neural operator baseline (Li et al., ICLR 2021)."""

    def __init__(self, in_channels: int, hidden: int, output_channels: int, modes: int = 12):
        super().__init__()
        width = max(24, hidden // 2)
        self.lift = torch.nn.Conv2d(in_channels, width, 1)
        self.spectral = torch.nn.ModuleList([SpectralConv2d(width, modes) for _ in range(4)])
        self.local = torch.nn.ModuleList([torch.nn.Conv2d(width, width, 1) for _ in range(4)])
        self.out = torch.nn.Sequential(torch.nn.Conv2d(width, width, 1), torch.nn.GELU(), torch.nn.Conv2d(width, output_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.lift(x)
        for spectral, local in zip(self.spectral, self.local, strict=True):
            feature = torch.nn.functional.gelu(spectral(feature) + local(feature))
        return self.out(feature)


class HaarWaveletBlock(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.mix = torch.nn.Conv2d(channels * 4, channels * 4, 1, groups=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_height, original_width = x.shape[-2:]
        if x.shape[-1] % 2 or x.shape[-2] % 2:
            x = torch.nn.functional.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2))
        a = x[:, :, 0::2, 0::2]
        b = x[:, :, 0::2, 1::2]
        c = x[:, :, 1::2, 0::2]
        d = x[:, :, 1::2, 1::2]
        bands = torch.cat([a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d], dim=1) * 0.5
        mixed = self.mix(bands)
        ll, lh, hl, hh = mixed.chunk(4, dim=1)
        output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        output[:, :, 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
        output[:, :, 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
        output[:, :, 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
        output[:, :, 1::2, 1::2] = (ll - lh - hl + hh) * 0.5
        return output[:, :, :original_height, :original_width]


class WNOOperator(torch.nn.Module):
    """Wavelet neural operator baseline following Tripura and Chakraborty (2023)."""

    def __init__(self, in_channels: int, hidden: int, output_channels: int):
        super().__init__()
        width = max(24, hidden // 2)
        self.lift = torch.nn.Conv2d(in_channels, width, 1)
        self.blocks = torch.nn.ModuleList([HaarWaveletBlock(width) for _ in range(4)])
        self.local = torch.nn.ModuleList([torch.nn.Conv2d(width, width, 1) for _ in range(4)])
        self.out = torch.nn.Conv2d(width, output_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.lift(x)
        for wavelet, local in zip(self.blocks, self.local, strict=True):
            feature = torch.nn.functional.gelu(wavelet(feature) + local(feature))
        return self.out(feature)
