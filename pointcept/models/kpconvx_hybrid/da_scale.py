class DensityAdaptiveScale(nn.Module):
    def __init__(self, s_min=0.5, s_max=2.0, eps=1e-6):
        super().__init__()
        self.s_min = s_min
        self.s_max = s_max
        self.eps = eps

    @torch.no_grad()
    def forward(self, points, neighbors, lengths=None):
        num_points = points.shape[0]
        valid = (neighbors >= 0) & (neighbors < num_points)
        safe_neighbors = neighbors.clamp(min=0, max=max(num_points - 1, 0))

        neigh_points = points[safe_neighbors]
        dist = torch.norm(neigh_points - points.unsqueeze(1), dim=-1)

        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_dist = (dist * valid.float()).sum(dim=1, keepdim=True) / denom

        rho = 1.0 / (mean_dist + self.eps)

        scale = torch.empty_like(rho)

        if lengths is None:
            rho_min = rho.min()
            rho_max = rho.max()
            rho_norm = (rho - rho_min) / (rho_max - rho_min + self.eps)
            scale[:] = self.s_min + (self.s_max - self.s_min) * (1.0 - rho_norm)
        else:
            start = 0
            for length in lengths.tolist():
                end = start + length
                rho_b = rho[start:end]
                rho_min = rho_b.min()
                rho_max = rho_b.max()
                rho_norm = (rho_b - rho_min) / (rho_max - rho_min + self.eps)
                scale[start:end] = self.s_min + (self.s_max - self.s_min) * (1.0 - rho_norm)
                start = end

        return scale.detach()
