import torch
import torch.nn as nn


class DAKPXBlockAdapter(nn.Module):
    """
    Stage-2 dynamic local adapter for KPConvX.

    Design principles:
        1. Do not rebuild graph / neighbor topology.
        2. Use existing pyramid neighbors from KPConvX.
        3. Estimate local density from current neighbors.
        4. Use two dynamic receptive-field branches:
            - small branch
            - large branch
        5. Fuse them with a learned gate.

    Input:
        feats     : [N, C]
        points    : [N, 3]
        neighbors : [N, K]

    Output:
        feats     : [N, C]
    """

    def __init__(
        self,
        dim,
        hidden_dim=None,
        dropout=0.0,
        scale_range=(0.75, 1.35),
        branch_scales=(0.85, 1.25),
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim

        self.dim = dim
        self.scale_min = float(scale_range[0])
        self.scale_max = float(scale_range[1])
        self.small_scale = float(branch_scales[0])
        self.large_scale = float(branch_scales[1])

        self.norm = nn.LayerNorm(dim)

        self.scale_mlp = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

        self.center_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.small_branch = nn.Sequential(
            nn.Linear(dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        self.large_branch = nn.Sequential(
            nn.Linear(dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _sanitize_neighbors(neighbors, num_points):
        valid = (neighbors >= 0) & (neighbors < num_points)
        safe_neighbors = neighbors.clamp(min=0, max=max(num_points - 1, 0))
        return safe_neighbors, valid.float()

    def _estimate_density(self, points, neighbors):
        """
        Density proxy:
            average distance to existing neighbors.
        """
        safe_neighbors, valid = self._sanitize_neighbors(neighbors, points.shape[0])

        neigh_points = points[safe_neighbors]          # [N, K, 3]
        center_points = points.unsqueeze(1)           # [N, 1, 3]
        dist = torch.norm(neigh_points - center_points, dim=-1)  # [N, K]

        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        density = (dist * valid).sum(dim=1, keepdim=True) / denom
        return density, dist, safe_neighbors, valid

    @staticmethod
    def _weighted_context(feats, safe_neighbors, dist, valid, effective_scale):
        """
        Weighted neighbor aggregation without changing topology.
        """
        neigh_feats = feats[safe_neighbors]  # [N, K, C]

        # effective_scale: [N, 1]
        effective_scale = effective_scale.clamp(min=1e-6)
        weight = torch.exp(-dist / effective_scale) * valid
        norm = weight.sum(dim=1, keepdim=True).clamp(min=1e-6)

        context = (weight.unsqueeze(-1) * neigh_feats).sum(dim=1) / norm
        return context

    def forward(self, feats, points, neighbors):
        if feats.numel() == 0:
            return feats

        x = self.norm(feats)

        density, dist, safe_neighbors, valid = self._estimate_density(points, neighbors)

        cond = torch.cat([x, density], dim=-1)

        scale = torch.sigmoid(self.scale_mlp(cond))
        scale = self.scale_min + (self.scale_max - self.scale_min) * scale  # [N, 1]

        gate = torch.softmax(self.gate_mlp(cond), dim=-1)  # [N, 2]

        center = self.center_proj(x)

        small_context = self._weighted_context(
            x,
            safe_neighbors,
            dist,
            valid,
            effective_scale=scale * self.small_scale,
        )

        large_context = self._weighted_context(
            x,
            safe_neighbors,
            dist,
            valid,
            effective_scale=scale * self.large_scale,
        )

        small_out = self.small_branch(
            torch.cat([center, small_context, density], dim=-1)
        )
        large_out = self.large_branch(
            torch.cat([center, large_context, density], dim=-1)
        )

        fused = gate[:, 0:1] * small_out + gate[:, 1:2] * large_out
        fused = self.out_proj(fused)

        return feats + fused
