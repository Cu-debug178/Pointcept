import torch
import torch.nn as nn


class DecoderRefineHead(nn.Module):
    """
    Lightweight decoder-side refinement head.

    Design principles:
        1. Reuse finest-level KPConvX neighbors.
        2. Do not rebuild graph / topology.
        3. Keep memory cost low for single 24G GPU training.
        4. Refine final decoder features before segmentation head.

    Input:
        feats     : [N, C]
        points    : [N, 3]
        neighbors : [N, K]

    Output:
        refined feats : [N, C]
    """

    def __init__(
        self,
        dim,
        hidden_dim=None,
        dropout=0.0,
        use_coords=True,
        use_boundary=True,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim

        self.dim = dim
        self.use_coords = use_coords
        self.use_boundary = use_boundary

        self.norm = nn.LayerNorm(dim)

        self.coord_proj = (
            nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )
            if use_coords
            else None
        )

        self.boundary_mlp = (
            nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
            if use_boundary
            else None
        )

        mix_in_dim = dim * 2 + 1 + (dim if use_coords else 0)
        gate_in_dim = dim * 2 + 1

        self.mix = nn.Sequential(
            nn.Linear(mix_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Sigmoid(),
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

    @staticmethod
    def _masked_mean(values, valid_mask):
        weight = valid_mask.unsqueeze(-1)
        denom = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (values * weight).sum(dim=1) / denom

    @staticmethod
    def _normalize_points(points):
        if points.numel() == 0:
            return points
        centered = points - points.mean(dim=0, keepdim=True)
        scale = centered.abs().amax(dim=0, keepdim=True).clamp(min=1e-6)
        return centered / scale

    def forward(self, feats, points, neighbors):
        if feats.numel() == 0:
            return feats

        x = self.norm(feats)
        safe_neighbors, valid = self._sanitize_neighbors(neighbors, feats.shape[0])

        neigh_feats = x[safe_neighbors]
        neigh_points = points[safe_neighbors]

        mean_feat = self._masked_mean(neigh_feats, valid)
        mean_point = self._masked_mean(neigh_points, valid)

        feat_diff = torch.norm(x - mean_feat, dim=-1, keepdim=True)
        coord_diff = torch.norm(points - mean_point, dim=-1, keepdim=True)

        if self.boundary_mlp is not None:
            boundary_score = self.boundary_mlp(torch.cat([feat_diff, coord_diff], dim=-1))
        else:
            boundary_score = feat_diff.new_ones((feat_diff.shape[0], 1))

        mix_inputs = [x, mean_feat, boundary_score]
        if self.coord_proj is not None:
            coord_embed = self.coord_proj(self._normalize_points(points))
            mix_inputs.append(coord_embed)

        refined = self.mix(torch.cat(mix_inputs, dim=-1))
        gate = self.gate(torch.cat([x, mean_feat, boundary_score], dim=-1))
        refined = self.out_proj(gate * refined)

        return feats + refined
