import torch
import torch.nn as nn


class DensityAdaptiveScale(nn.Module):
    """
    Density-to-scale mapper for DA-KPConvX.

    rho_i = 1 / (mean KNN distance + eps)

    Normalized linear mapping:

        s_i = s_min + (s_max - s_min) * (1 - norm(rho_i))

    Dense area:
        rho high -> s_i close to s_min

    Sparse area:
        rho low -> s_i close to s_max
    """

    def __init__(self, scale_range=(0.5, 2.0), density_k=16, eps=1e-6):
        super().__init__()
        self.s_min = float(scale_range[0])
        self.s_max = float(scale_range[1])
        self.density_k = int(density_k) if density_k is not None else None
        self.eps = float(eps)

        if self.s_min <= 0 or self.s_max <= 0 or self.s_min > self.s_max:
            raise ValueError(f"Invalid scale_range: {scale_range}")

    @staticmethod
    def _sanitize_neighbors(neighbors, num_points):
        valid = (neighbors >= 0) & (neighbors < num_points)
        safe_neighbors = neighbors.clamp(min=0, max=max(num_points - 1, 0))
        return safe_neighbors, valid.float()

    def _map_density_to_scale(self, rho, lengths=None):
        scale = torch.empty_like(rho)

        if lengths is None:
            rho_min = rho.min()
            rho_max = rho.max()
            rho_norm = (rho - rho_min) / (rho_max - rho_min + self.eps)
            scale = self.s_min + (self.s_max - self.s_min) * (1.0 - rho_norm)
            return scale.clamp(min=self.s_min, max=self.s_max)

        start = 0
        for length in lengths.tolist():
            end = start + int(length)
            if end <= start:
                start = end
                continue

            rho_b = rho[start:end]
            rho_min = rho_b.min()
            rho_max = rho_b.max()
            rho_norm = (rho_b - rho_min) / (rho_max - rho_min + self.eps)

            scale[start:end] = self.s_min + (self.s_max - self.s_min) * (1.0 - rho_norm)
            start = end

        return scale.clamp(min=self.s_min, max=self.s_max)

    @torch.no_grad()
    def forward(self, points, neighbors, lengths=None):
        if points.numel() == 0:
            return points.new_zeros((0, 1))

        if self.density_k is not None and neighbors.shape[1] > self.density_k:
            neighbors = neighbors[:, : self.density_k]

        safe_neighbors, valid = self._sanitize_neighbors(neighbors, points.shape[0])

        neigh_points = points[safe_neighbors]
        dist = torch.norm(neigh_points - points.unsqueeze(1), dim=-1)

        # Remove self-neighbor with zero distance.
        self_mask = dist <= self.eps
        valid = valid * (~self_mask).float()

        valid_count = valid.sum(dim=1, keepdim=True)
        mean_dist = (dist * valid).sum(dim=1, keepdim=True) / valid_count.clamp(min=1.0)

        # If a point has no valid non-self neighbors, use a safe fallback.
        no_neighbor = valid_count <= 0
        if no_neighbor.any():
            if (~no_neighbor).any():
                fallback_dist = mean_dist[~no_neighbor].mean()
            else:
                fallback_dist = points.new_tensor(1.0)

            mean_dist = torch.where(
                no_neighbor,
                fallback_dist.expand_as(mean_dist),
                mean_dist,
            )

        rho = 1.0 / (mean_dist + self.eps)

        da_scale = self._map_density_to_scale(rho, lengths=lengths)
        return da_scale.detach()
