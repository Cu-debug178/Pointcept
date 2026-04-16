import torch
import torch.nn as nn


class DecoderRefineModule(nn.Module):
    """
    Lightweight refine module placed after decoder and before segmentation head.

    Design:
        - keep the channel size unchanged
        - reuse finest-level neighbors already built by KPConvX pyramid
        - enhance boundary/detail recovery with local residual refinement
    """

    def __init__(
        self,
        dim,
        hidden_dim=None,
        dropout=0.0,
        use_coord=True,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.use_coord = use_coord

        self.norm = nn.LayerNorm(dim)
        self.coord_proj = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ) if use_coord else None

        self.edge_mlp = nn.Sequential(
            nn.Linear(dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.refine_mlp = nn.Sequential(
            nn.Linear(dim * 3 + (3 if use_coord else 0), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
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

    def forward(self, feats, points, neighbors):
        if feats.numel() == 0:
            return feats

        x = self.norm(feats)
        safe_neighbors, valid = self._sanitize_neighbors(neighbors, x.shape[0])

        neigh_feats = x[safe_neighbors]                              # [N, K, C]
        center_feats = x.unsqueeze(1).expand_as(neigh_feats)         # [N, K, C]

        neigh_points = points[safe_neighbors]                        # [N, K, 3]
        center_points = points.unsqueeze(1)                          # [N, 1, 3]
        rel_points = neigh_points - center_points                    # [N, K, 3]
        dist = torch.norm(rel_points, dim=-1, keepdim=True)          # [N, K, 1]

        edge_logits = self.edge_mlp(
            torch.cat([center_feats, neigh_feats, dist], dim=-1)
        ).squeeze(-1)
        edge_logits = edge_logits.masked_fill(valid <= 0, float("-inf"))
        edge_weight = torch.softmax(edge_logits, dim=1)
        edge_weight = edge_weight * valid
        edge_norm = edge_weight.sum(dim=1, keepdim=True).clamp(min=1e-6)
        edge_weight = edge_weight / edge_norm

        context = (edge_weight.unsqueeze(-1) * neigh_feats).sum(dim=1)
        detail = x - context

        refine_inputs = [x, context, detail]
        if self.use_coord:
            coord_embed = self.coord_proj(rel_points.mean(dim=1))
            refine_inputs.append(coord_embed)

        refined = self.refine_mlp(torch.cat(refine_inputs, dim=-1))
        refined = self.out_proj(refined)
        return feats + refined
