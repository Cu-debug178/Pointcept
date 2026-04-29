import torch
import torch.nn as nn


class GeometryDifficultyRouter(nn.Module):
    """
    Lightweight geometry-difficulty router.

    Goals:
        1. Reuse existing KPConvX neighborhood topology.
        2. Predict a difficulty score per point.
        3. Control how much SGCA residual is injected.

    Outputs:
        difficulty    : [N, 1] in [0, 1]
        global_weight : [N, 1] router weight for SGCA residual
    """

    def __init__(
        self,
        dim,
        hidden_dim=None,
        dropout=0.0,
        temperature=1.0,
        global_boost=1.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or max(32, dim // 2)

        self.dim = dim
        self.temperature = max(float(temperature), 1e-6)
        self.global_boost = float(global_boost)

        self.norm = nn.LayerNorm(dim)

        self.router_mlp = nn.Sequential(
            nn.Linear(dim + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.difficulty_head = nn.Linear(hidden_dim, 1)

        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _sanitize_neighbors(neighbors, num_points):
        valid = (neighbors >= 0) & (neighbors < num_points)
        safe_neighbors = neighbors.clamp(min=0, max=max(num_points - 1, 0))
        return safe_neighbors, valid.float()

    @staticmethod
    def _masked_mean(values, valid_mask):
        denom = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (values * valid_mask).sum(dim=1, keepdim=True) / denom

    def _collect_stats(self, feats, points, neighbors):
        x = self.norm(feats)
        safe_neighbors, valid = self._sanitize_neighbors(neighbors, feats.shape[0])

        neigh_points = points[safe_neighbors]
        center_points = points.unsqueeze(1)
        dist = torch.norm(neigh_points - center_points, dim=-1)

        mean_dist = self._masked_mean(dist, valid)
        dist_centered = dist - mean_dist
        dist_var = self._masked_mean(dist_centered * dist_centered, valid)

        neigh_feats = x[safe_neighbors]
        feat_dist = torch.norm(neigh_feats - x.unsqueeze(1), dim=-1)
        feat_var = self._masked_mean(feat_dist, valid)

        return x, mean_dist, dist_var, feat_var

    def forward(self, feats, points, neighbors):
        if feats.numel() == 0:
            empty_1 = feats.new_zeros((0, 1))
            return {
                "difficulty": empty_1,
                "global_weight": empty_1,
            }

        x, mean_dist, dist_var, feat_var = self._collect_stats(
            feats,
            points,
            neighbors,
        )

        router_input = torch.cat(
            [x, mean_dist, dist_var, feat_var],
            dim=-1,
        )

        hidden = self.router_mlp(router_input)

        difficulty_logit = self.difficulty_head(hidden) / self.temperature
        difficulty = torch.sigmoid(difficulty_logit)

        global_logit = self.global_head(
            torch.cat([hidden, difficulty], dim=-1)
        )

        global_weight = torch.sigmoid(
            global_logit + self.global_boost * difficulty_logit
        )

        return {
            "difficulty": difficulty,
            "global_weight": global_weight,
        }
