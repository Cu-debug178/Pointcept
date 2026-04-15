import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def offset_to_lengths(offset: torch.Tensor) -> torch.Tensor:
    offset = offset.int()
    offset0 = torch.cat(
        [torch.zeros(1, device=offset.device, dtype=offset.dtype), offset], dim=0
    )
    return offset0[1:] - offset0[:-1]


def lengths_to_batch_index(lengths: torch.Tensor) -> torch.Tensor:
    batch_ids = []
    for b, ln in enumerate(lengths.tolist()):
        batch_ids.append(
            torch.full((ln,), b, device=lengths.device, dtype=torch.long)
        )
    return torch.cat(batch_ids, dim=0)


def batch_segment_mean(x: torch.Tensor, batch_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
    out = x.new_zeros((batch_size, x.shape[-1]))
    cnt = x.new_zeros((batch_size, 1))
    out.index_add_(0, batch_ids, x)
    cnt.index_add_(0, batch_ids, torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype))
    out = out / cnt.clamp(min=1.0)
    return out


class FeedForward(nn.Module):
    def __init__(self, channels: int, hidden_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(channels * hidden_ratio)
        self.fc1 = nn.Linear(channels, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SGCAttentionBlock(nn.Module):
    """
    Lite serialized global context block.

    Input:
        x: [M, C]
    Output:
        x: [M, C]
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop_path1 = nn.Identity()
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = FeedForward(channels, hidden_ratio=mlp_ratio, drop=proj_drop)
        self.drop_path2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, M, C]
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_path1(h)
        x = x + self.drop_path2(self.ffn(self.norm2(x)))
        return x


class GlobalContextFusion(nn.Module):
    """
    Fuse local features with stage-level global token.
    """

    def __init__(self, channels: int, token_channels: int = None):
        super().__init__()
        token_channels = token_channels or channels
        self.token_proj = nn.Linear(token_channels, channels)
        self.gate = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # x: [N, C], g: [N, C]
        gp = self.token_proj(g)
        alpha = self.gate(torch.cat([x, gp], dim=-1))
        fused = torch.cat([x, alpha * gp], dim=-1)
        return x + self.out_proj(fused)


class SGCALite(nn.Module):
    """
    SGCA-lite:
    1) serialize points with a lightweight order
    2) split into patches
    3) apply patch-wise MHSA
    4) produce a stage global token
    5) fuse token back to each point

    This is intentionally simpler than full PTv3:
    - no FlashAttention requirement
    - no heavy Point serialization dependency
    - easy to insert into existing KPConvX hybrid branch
    """

    def __init__(
        self,
        channels: int,
        patch_size: int = 128,
        num_heads: int = 8,
        num_blocks: int = 1,
        mlp_ratio: float = 4.0,
        order_mode: str = "xyz",
        with_pos_embed: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.order_mode = order_mode
        self.with_pos_embed = with_pos_embed

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        ) if with_pos_embed else None

        self.blocks = nn.ModuleList(
            [
                SGCAttentionBlock(
                    channels=channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_blocks)
            ]
        )

        self.fusion = GlobalContextFusion(channels)
        self.token_norm = nn.LayerNorm(channels)

    @staticmethod
    def _normalize_xyz(xyz: torch.Tensor) -> torch.Tensor:
        xyz_min = xyz.min(dim=0, keepdim=True)[0]
        xyz_max = xyz.max(dim=0, keepdim=True)[0]
        scale = (xyz_max - xyz_min).clamp(min=1e-6)
        return (xyz - xyz_min) / scale

    def _serialize_single_cloud(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Return a permutation index for one cloud.
        A lite replacement for PTv3 shuffle-order serialization.
        """
        xyz_n = self._normalize_xyz(xyz)

        x = xyz_n[:, 0]
        y = xyz_n[:, 1]
        z = xyz_n[:, 2]

        if self.order_mode == "xyz":
            score = x * 1e6 + y * 1e3 + z
        elif self.order_mode == "xzy":
            score = x * 1e6 + z * 1e3 + y
        elif self.order_mode == "zxy":
            score = z * 1e6 + x * 1e3 + y
        elif self.order_mode == "hilbert_like":
            score = (
                (x + y) * 1e6
                + (y + z) * 1e3
                + (z + x)
            )
        else:
            score = x * 1e6 + y * 1e3 + z

        return torch.argsort(score)

    def _build_patches(
        self,
        coords: torch.Tensor,
        lengths: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Return a list of index tensors, one tensor per patch.
        """
        patch_indices: List[torch.Tensor] = []
        start = 0
        for ln in lengths.tolist():
            end = start + ln
            xyz = coords[start:end]
            order = self._serialize_single_cloud(xyz)
            ordered_idx = torch.arange(start, end, device=coords.device)[order]

            for p_start in range(0, ln, self.patch_size):
                p_end = min(p_start + self.patch_size, ln)
                patch_indices.append(ordered_idx[p_start:p_end])

            start = end
        return patch_indices

    def _run_patch_attention(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        patch_indices: List[torch.Tensor],
    ) -> torch.Tensor:
        out = x.clone()
        for idx in patch_indices:
            patch_x = x[idx]  # [M, C]
            if self.with_pos_embed:
                patch_pos = self._normalize_xyz(coords[idx])
                patch_x = patch_x + self.pos_mlp(patch_pos)

            patch_x = patch_x.unsqueeze(0)  # [1, M, C]
            for blk in self.blocks:
                patch_x = blk(patch_x)
            out[idx] = patch_x.squeeze(0)
        return out

    def forward(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        offset: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feats:  [N, C]
            coords: [N, 3]
            offset: [B]
        Returns:
            fused_feats: [N, C]
            point_global_token: [N, C]
        """
        lengths = offset_to_lengths(offset)
        batch_ids = lengths_to_batch_index(lengths)

        patch_indices = self._build_patches(coords, lengths)
        ctx_feats = self._run_patch_attention(feats, coords, patch_indices)

        batch_global = batch_segment_mean(ctx_feats, batch_ids, batch_size=len(lengths))
        batch_global = self.token_norm(batch_global)
        point_global = batch_global[batch_ids]

        fused = self.fusion(feats, point_global)
        return fused, point_global
