import torch
import torch.nn as nn


class DAKPXBlockAdapter(nn.Module):
    """
    Fine / Coarse dual-scale local adapter for KPConvX.

    Design goals
    ------------
    1. Reuse KPConvX pyramid neighbors, no graph rebuild.
    2. Keep output channel unchanged.
    3. Make the two local branches explicit:
        - fine branch   : smaller effective receptive field
        - coarse branch : larger effective receptive field
    4. Use a lightweight density-aware gate for adaptive fusion.
    5. Keep memory/computation friendly for 24 GB GPU training.

    Input
    -----
    feats     : [N, C]
    points    : [N, 3]
    neighbors : [N, K]

    Output
    ------
    feats     : [N, C]
    """

    def __init__(
        self,
        dim,
        hidden_dim=None,
        hidden_ratio=0.75,
        dropout=0.0,
        scale_range=(0.80, 1.25),
        fine_scale=0.85,
        coarse_scale=1.20,
        use_density=True,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = max(dim, int(round(dim * hidden_ratio)))

        self.dim = dim
        self.use_density = use_density
        self.scale_min = float(scale_range[0])
        self.scale_max = float(scale_range[1])
        self.fine_scale = float(fine_scale)
        self.coarse_scale = float(coarse_scale)

        self.norm = nn.LayerNorm(dim)

        cond_dim = dim + (2 if use_density else 1)

        self.scale_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

        self.center_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        branch_in_dim = dim * 3 + (2 if use_density else 1)

        self.fine_branch = nn.Sequential(
            nn.Linear(branch_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        self.coarse_branch = nn.Sequential(
            nn.Linear(branch_in_dim, hidden_dim),
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

    def _estimate_local_stats(self, points, neighbors):
        """
        Return:
            density_token : [N, 2] if use_density else [N, 1]
                first channel  -> average neighbor distance
                second channel -> valid neighbor ratio
            dist          : [N, K]
            safe_neighbors: [N, K]
            valid         : [N, K]
        """
        safe_neighbors, valid = self._sanitize_neighbors(neighbors, points.shape[0])

        neigh_points = points[safe_neighbors]                # [N, K, 3]
        center_points = points.unsqueeze(1)                  # [N, 1, 3]
        rel = neigh_points - center_points                   # [N, K, 3]
        dist = torch.norm(rel, dim=-1)                       # [N, K]

        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        avg_dist = (dist * valid).sum(dim=1, keepdim=True) / denom
        valid_ratio = valid.mean(dim=1, keepdim=True)

        if self.use_density:
            density_token = torch.cat([avg_dist, valid_ratio], dim=-1)
        else:
            density_token = avg_dist

        return density_token, dist, safe_neighbors, valid

    @staticmethod
    def _weighted_context(feats, safe_neighbors, dist, valid, effective_scale):
        """
        Neighbor aggregation with adaptive effective scale.
        """
        neigh_feats = feats[safe_neighbors]                  # [N, K, C]
        effective_scale = effective_scale.clamp(min=1e-6)   # [N, 1]
        weight = torch.exp(-dist / effective_scale) * valid
        norm = weight.sum(dim=1, keepdim=True).clamp(min=1e-6)
        context = (weight.unsqueeze(-1) * neigh_feats).sum(dim=1) / norm
        return context

    def forward(self, feats, points, neighbors):
        if feats.numel() == 0:
            return feats

        x = self.norm(feats)

        density_token, dist, safe_neighbors, valid = self._estimate_local_stats(
            points, neighbors
        )

        cond = torch.cat([x, density_token], dim=-1)

        dyn_scale = torch.sigmoid(self.scale_mlp(cond))
        dyn_scale = self.scale_min + (self.scale_max - self.scale_min) * dyn_scale

        gate = torch.softmax(self.gate_mlp(cond), dim=-1)   # [N, 2]

        center = self.center_proj(x)

        fine_context = self._weighted_context(
            x,
            safe_neighbors,
            dist,
            valid,
            effective_scale=dyn_scale * self.fine_scale,
        )
        coarse_context = self._weighted_context(
            x,
            safe_neighbors,
            dist,
            valid,
            effective_scale=dyn_scale * self.coarse_scale,
        )

        fine_detail = center - fine_context
        coarse_detail = center - coarse_context

        fine_out = self.fine_branch(
            torch.cat([center, fine_context, fine_detail, density_token], dim=-1)
        )
        coarse_out = self.coarse_branch(
            torch.cat([center, coarse_context, coarse_detail, density_token], dim=-1)
        )

        fused = gate[:, 0:1] * fine_out + gate[:, 1:2] * coarse_out
        fused = self.out_proj(fused)

        return feats + fused
