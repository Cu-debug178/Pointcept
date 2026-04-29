#
# Copied and modified from KPConvX project: kpnext_blocks.py
#
# DA-KPConvX version:
#   1. Keep baseline KPConvX untouched.
#   2. Add density-adaptive kernel point scaling to KPConvX only.
#   3. Use da_scale to scale kernel points:
#          p_k_da = s_i * p_k
#   4. Do not reuse influence cache when da_scale is provided.
#

import math
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from torch.nn.init import kaiming_uniform_

from pointcept.models.kpconvx.utils.kernel_points import load_kernels
from pointcept.models.kpconvx.utils.generic_blocks import (
    gather,
    index_select,
    radius_gaussian,
    local_maxpool,
    UnaryBlock,
    NormBlock,
    DropPathPack,
    build_mlp,
    mlp_from_list,
)


# ----------------------------------------------------------------------------------------------------------------------
#
#           KPConvD Class
#       \*******************/
#


class KPConvD(nn.Module):
    """
    Original KPConvD copied from baseline.

    Important:
        DA-KPConvX does not modify KPConvD.
        Density-adaptive kernel scaling is only applied to DAKPConvX.
    """

    def __init__(
        self,
        channels: int,
        shell_sizes: list,
        radius: float,
        sigma: float,
        Cmid: int = 0,
        shared_kp_data=None,
        dimension: int = 3,
        influence_mode: str = "linear",
        fixed_kernel_points: str = "center",
        norm_type: str = "batch",
        bn_momentum: float = 0.1,
        activation: nn.Module = nn.LeakyReLU(0.1),
        inf: float = 1e6,
    ):
        super(KPConvD, self).__init__()

        self.channels = channels
        self.shell_sizes = shell_sizes
        self.K = int(np.sum(shell_sizes))
        self.radius = radius
        self.sigma = sigma
        self.dimension = dimension
        self.influence_mode = influence_mode
        self.fixed_kernel_points = fixed_kernel_points
        self.inf = inf
        self.Cmid = Cmid

        if Cmid > 0:
            self.weights = nn.Parameter(torch.zeros(size=(self.K, Cmid)), requires_grad=True)
            self.out_mlp = nn.Linear(Cmid * channels, channels)
        else:
            self.weights = nn.Parameter(torch.zeros(size=(self.K, channels)), requires_grad=True)

        self.reset_parameters()

        self.share_kp = shared_kp_data is not None
        self.first_kp = False

        if self.share_kp:
            self.first_kp = "k_pts" not in shared_kp_data
            self.shared_kp_data = shared_kp_data
            if self.first_kp:
                self.shared_kp_data["k_pts"] = self.initialize_kernel_points()
            self.register_buffer("kernel_points", self.shared_kp_data["k_pts"])
        else:
            self.shared_kp_data = {}
            kernel_points = self.initialize_kernel_points()
            self.register_buffer("kernel_points", kernel_points)
            self.shared_kp_data["k_pts"] = kernel_points

        self.merge_op = torch.mul
        self.aggr_op = torch.sum

        if self.influence_mode == "mlp":
            if Cmid > 0:
                self.delta_mlp = UnaryBlock(
                    self.dimension,
                    Cmid,
                    norm_type,
                    bn_momentum,
                    activation,
                )
            else:
                self.delta_mlp = build_mlp(
                    n_layers=2,
                    Cin=self.dimension,
                    Cmid=16,
                    Cout=channels,
                    norm_type=norm_type,
                    bn_momentum=bn_momentum,
                    activation=activation,
                )

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))

    def initialize_kernel_points(self) -> Tensor:
        kernel_points = load_kernels(
            self.radius,
            self.shell_sizes,
            dimension=self.dimension,
            fixed=self.fixed_kernel_points,
        )
        return torch.from_numpy(kernel_points).float()

    @torch.no_grad()
    def get_neighbors_influences(
        self,
        q_pts: Tensor,
        s_pts: Tensor,
        neighb_inds: Tensor,
    ) -> Tensor:
        """
        Original KPConvD influence computation.
        No da_scale here.
        """

        if self.share_kp and not self.first_kp:
            influence_weights = self.shared_kp_data["infl_w"]
            neighbors = self.shared_kp_data["neighb_p"]
            neighbors_1nn = self.shared_kp_data["neighb_1nn"]

        else:
            s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + self.inf), 0)

            neighbors = index_select(s_pts, neighb_inds, dim=0)
            neighbors = neighbors - q_pts.unsqueeze(1)

            if self.influence_mode == "mlp":
                neighbors *= 1 / self.radius
                neighbors_1nn = None
                influence_weights = None

            else:
                differences = neighbors.unsqueeze(2) - self.kernel_points
                sq_distances = torch.sum(differences ** 2, dim=3)

                nn_sq_dists, neighbors_1nn = torch.min(sq_distances, dim=2)

                influence_weights = None
                if self.influence_mode != "constant":
                    if self.influence_mode == "linear":
                        influence_weights = torch.clamp(
                            1 - torch.sqrt(nn_sq_dists) / self.sigma,
                            min=0.0,
                        )
                    elif self.influence_mode == "gaussian":
                        gaussian_sigma = self.sigma * 0.3
                        influence_weights = radius_gaussian(nn_sq_dists, gaussian_sigma)
                    else:
                        raise ValueError(
                            "Unknown influence mode: '{:s}'. Should be 'constant', 'linear', or 'gaussian'".format(
                                self.influence_mode
                            )
                        )

            if self.share_kp:
                self.shared_kp_data["neighb_1nn"] = neighbors_1nn
                self.shared_kp_data["neighb_p"] = neighbors
                self.shared_kp_data["infl_w"] = influence_weights

        return influence_weights, neighbors, neighbors_1nn

    def forward(
        self,
        q_pts: Tensor,
        s_pts: Tensor,
        s_feats: Tensor,
        neighb_inds: Tensor,
    ) -> Tensor:
        padded_s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)
        neighbor_feats = index_select(padded_s_feats, neighb_inds, dim=0)

        influence_weights, neighbors, neighbors_1nn = self.get_neighbors_influences(
            q_pts,
            s_pts,
            neighb_inds,
        )

        if self.influence_mode == "mlp":
            neighbors_weights = self.delta_mlp(neighbors)
        else:
            neighbors_weights = gather(self.weights, neighbors_1nn)

            if self.influence_mode != "constant":
                neighbors_weights *= influence_weights.unsqueeze(2)

        if self.Cmid > 0:
            intermediate_feats = torch.matmul(
                neighbors_weights.transpose(1, 2),
                neighbor_feats,
            )
            output_feats = self.out_mlp(
                intermediate_feats.reshape(-1, self.Cmid * self.channels)
            )
        else:
            output_feats = self.aggr_op(
                self.merge_op(neighbor_feats, neighbors_weights),
                dim=1,
            )

        return output_feats

    def __repr__(self):
        repr_str = "KPConvD"
        repr_str += "(K: {:d}".format(self.K)
        repr_str += ", C: {:d}".format(self.channels)
        repr_str += ", r: {:.2f}".format(self.radius)
        repr_str += ", sigma: {:.2f})".format(self.sigma)
        return repr_str


# ----------------------------------------------------------------------------------------------------------------------
#
#           DA-KPConvX Class
#       \*******************/
#


class DAKPConvX(nn.Module):
    """
    Density-Adaptive KPConvX.

    Difference from original KPConvX:
        Original:
            differences = neighbors - kernel_points

        DA version:
            scaled_kernel_points_i = da_scale_i * kernel_points
            differences = neighbors - scaled_kernel_points_i

    This implements:

        p_tilde_k = s_i * p_k

    where:
        s_i < 1 for dense regions
        s_i > 1 for sparse regions
    """

    def __init__(
        self,
        channels: int,
        shell_sizes: list,
        radius: float,
        sigma: float,
        attention_groups: int = 8,
        attention_act: str = "sigmoid",
        mod_grp_norm: bool = False,
        shared_kp_data=None,
        dimension: int = 3,
        influence_mode: str = "linear",
        fixed_kernel_points: str = "center",
        norm_type: str = "batch",
        bn_momentum: float = 0.1,
        activation: nn.Module = nn.LeakyReLU(0.1),
        inf: float = 1e6,
    ):
        super(DAKPConvX, self).__init__()

        if attention_groups > 0:
            assert channels % attention_groups == 0, "channels must be divisible by attention_groups."
            ch_per_grp = channels // attention_groups
        else:
            ch_per_grp = -attention_groups
            assert channels % ch_per_grp == 0, "channels must be divisible by ch_per_grp."
            attention_groups = channels // ch_per_grp

        self.channels = channels
        self.shell_sizes = shell_sizes
        self.K = int(np.sum(shell_sizes))
        self.radius = radius
        self.sigma = sigma
        self.dimension = dimension
        self.influence_mode = influence_mode
        self.fixed_kernel_points = fixed_kernel_points
        self.inf = inf
        self.ch_per_grp = ch_per_grp
        self.groups = attention_groups
        self.mod_grp_norm = mod_grp_norm

        self.weights = nn.Parameter(torch.zeros(size=(self.K, channels)), requires_grad=True)
        kaiming_uniform_(self.weights, a=math.sqrt(5))

        self.share_kp = shared_kp_data is not None
        self.first_kp = False

        if self.share_kp:
            self.first_kp = "k_pts" not in shared_kp_data
            self.shared_kp_data = shared_kp_data
            if self.first_kp:
                self.shared_kp_data["k_pts"] = self.initialize_kernel_points()
            self.register_buffer("kernel_points", self.shared_kp_data["k_pts"])
        else:
            self.shared_kp_data = {}
            kernel_points = self.initialize_kernel_points()
            self.register_buffer("kernel_points", kernel_points)
            self.shared_kp_data["k_pts"] = kernel_points

        self.merge_op = torch.mul
        self.aggr_op = torch.sum

        Cout = self.K * self.ch_per_grp
        alpha_list = [channels, "NA", Cout]

        self.alpha_mlp = mlp_from_list(
            channels,
            alpha_list,
            final_bias=False,
            norm_type="none",
            bn_momentum=-1,
            activation=activation,
        )

        self.grpnorm = nn.GroupNorm(self.K, self.K * self.ch_per_grp)

        if attention_act == "sigmoid":
            self.attention_act = torch.sigmoid
        elif attention_act == "tanh":
            self.attention_act = torch.tanh
        elif attention_act == "softmax":
            self.attention_act = nn.Softmax(dim=1)
        else:
            self.attention_act = nn.Identity()

    def initialize_kernel_points(self) -> Tensor:
        kernel_points = load_kernels(
            self.radius,
            self.shell_sizes,
            dimension=self.dimension,
            fixed=self.fixed_kernel_points,
        )
        return torch.from_numpy(kernel_points).float()

    @torch.no_grad()
    def get_neighbors_influences(
        self,
        q_pts: Tensor,
        s_pts: Tensor,
        neighb_inds: Tensor,
        da_scale: Tensor = None,
    ) -> Tensor:
        """
        Influence function of density-adaptive kernel points on neighbors.

        Args:
            q_pts: query points, shape [M, 3]
            s_pts: support points, shape [N, 3]
            neighb_inds: neighbor indices, shape [M, H]
            da_scale: adaptive scale for each query point, shape [M] or [M, 1]
        """

        # When da_scale is provided, influence depends on each query point.
        # Therefore cached influence weights cannot be reused.
        if da_scale is None and self.share_kp and not self.first_kp:
            influence_weights = self.shared_kp_data["infl_w"]
            neighbors = self.shared_kp_data["neighb_p"]
            neighbors_1nn = self.shared_kp_data["neighb_1nn"]

        else:
            s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + self.inf), 0)

            neighbors = index_select(s_pts, neighb_inds, dim=0)
            neighbors = neighbors - q_pts.unsqueeze(1)

            if da_scale is not None:
                da_scale = da_scale.to(device=neighbors.device, dtype=neighbors.dtype).view(-1, 1, 1)

                if da_scale.shape[0] != neighbors.shape[0]:
                    raise ValueError(
                        f"da_scale length {da_scale.shape[0]} does not match query points {neighbors.shape[0]}"
                    )

                scaled_kernel_points = self.kernel_points.unsqueeze(0) * da_scale
                differences = neighbors.unsqueeze(2) - scaled_kernel_points.unsqueeze(1)
            else:
                differences = neighbors.unsqueeze(2) - self.kernel_points

            sq_distances = torch.sum(differences ** 2, dim=3)

            nn_sq_dists, neighbors_1nn = torch.min(sq_distances, dim=2)

            influence_weights = None
            if self.influence_mode != "constant":
                if self.influence_mode == "linear":
                    influence_weights = torch.clamp(
                        1 - torch.sqrt(nn_sq_dists) / self.sigma,
                        min=0.0,
                    )
                elif self.influence_mode == "gaussian":
                    gaussian_sigma = self.sigma * 0.3
                    influence_weights = radius_gaussian(nn_sq_dists, gaussian_sigma)
                else:
                    raise ValueError(
                        "Unknown influence mode: '{:s}'. Should be 'constant', 'linear', or 'gaussian'".format(
                            self.influence_mode
                        )
                    )

            # Cache only in original mode.
            # In DA mode, influence weights depend on da_scale and must be recomputed.
            if da_scale is None and self.share_kp:
                self.shared_kp_data["neighb_1nn"] = neighbors_1nn
                self.shared_kp_data["neighb_p"] = neighbors
                self.shared_kp_data["infl_w"] = influence_weights

        return influence_weights, neighbors, neighbors_1nn

    def forward(
        self,
        q_pts: Tensor,
        s_pts: Tensor,
        s_feats: Tensor,
        neighb_inds: Tensor,
        da_scale: Tensor = None,
    ) -> Tensor:
        """
        DA-KPConvX forward.
        """

        padded_s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)
        neighbor_feats = index_select(padded_s_feats, neighb_inds, dim=0)

        if q_pts.shape[0] == s_pts.shape[0]:
            pooled_feats = s_feats
        else:
            pooled_feats = neighbor_feats[:, 0, :]

        modulations = self.alpha_mlp(pooled_feats)

        if self.mod_grp_norm:
            modulations = modulations.transpose(0, 1).unsqueeze(0)
            modulations = self.grpnorm(modulations)
            modulations = modulations.squeeze(0).transpose(0, 1)

        modulations = self.attention_act(modulations)

        modulations = modulations.view(-1, self.K, self.ch_per_grp, 1)
        conv_weights = self.weights.view(1, self.K, self.ch_per_grp, self.groups)

        conv_weights = conv_weights * modulations
        conv_weights = conv_weights.reshape(-1, self.K, self.channels)

        influence_weights, neighbors, neighbors_1nn = self.get_neighbors_influences(
            q_pts,
            s_pts,
            neighb_inds,
            da_scale=da_scale,
        )

        neighbors_weights = torch.gather(
            conv_weights,
            1,
            neighbors_1nn.unsqueeze(2).expand(-1, -1, self.channels),
        )

        if self.influence_mode != "constant":
            neighbors_weights *= influence_weights.unsqueeze(2)

        neighbor_feats = self.merge_op(neighbor_feats, neighbors_weights)

        output_feats = self.aggr_op(neighbor_feats, dim=1)

        return output_feats

    def __repr__(self):
        repr_str = "DAKPConvX"
        repr_str += "(K: {:d}".format(self.K)
        repr_str += ", C: {:d}".format(self.channels)
        repr_str += ", r: {:.2f}".format(self.radius)
        repr_str += ", sigma: {:.2f})".format(self.sigma)
        return repr_str


# ----------------------------------------------------------------------------------------------------------------------
#
#           Network blocks
#       \********************/
#


class KPNextBlock(nn.Module):
    """
    Optional copied block.

    This block is kept for compatibility.
    DA is passed only when the convolution is DAKPConvX.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shell_sizes: list,
        radius: float,
        sigma: float,
        attention_groups: int = 8,
        attention_act: str = "sigmoid",
        mod_grp_norm: bool = False,
        mlp_first: bool = True,
        shared_kp_data=None,
        influence_mode: str = "linear",
        dimension: int = 3,
        norm_type: str = "batch",
        bn_momentum: float = 0.1,
        activation: nn.Module = nn.LeakyReLU(0.1),
    ):
        super(KPNextBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.influence_mode = influence_mode
        self.radius = radius
        self.bn_momentum = bn_momentum
        self.norm_type = norm_type
        self.mlp_first = mlp_first

        if mlp_first:
            conv_channels = out_channels
        else:
            conv_channels = in_channels

        self.activation = activation
        self.conv_norm = NormBlock(conv_channels, norm_type, bn_momentum)

        if attention_groups == 0:
            self.conv = KPConvD(
                conv_channels,
                shell_sizes,
                radius,
                sigma,
                Cmid=0,
                shared_kp_data=shared_kp_data,
                dimension=dimension,
                influence_mode=influence_mode,
                norm_type=norm_type,
                bn_momentum=bn_momentum,
                activation=activation,
            )
        else:
            self.conv = DAKPConvX(
                conv_channels,
                shell_sizes,
                radius,
                sigma,
                attention_groups=attention_groups,
                attention_act=attention_act,
                mod_grp_norm=mod_grp_norm,
                shared_kp_data=shared_kp_data,
                dimension=dimension,
                influence_mode=influence_mode,
                norm_type=norm_type,
                bn_momentum=bn_momentum,
                activation=activation,
            )

        if in_channels != out_channels:
            self.up_mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                NormBlock(out_channels, norm_type, bn_momentum),
                activation,
            )

    def forward(self, q_pts, s_pts, x, neighbor_indices, da_scale=None):
        if self.mlp_first:
            if self.in_channels != self.out_channels:
                x = self.up_mlp(x)

            if isinstance(self.conv, DAKPConvX):
                x = self.conv(q_pts, s_pts, x, neighbor_indices, da_scale=da_scale)
            else:
                x = self.conv(q_pts, s_pts, x, neighbor_indices)

            x = self.conv_norm(x)
            x = self.activation(x)

        else:
            if isinstance(self.conv, DAKPConvX):
                x = self.conv(q_pts, s_pts, x, neighbor_indices, da_scale=da_scale)
            else:
                x = self.conv(q_pts, s_pts, x, neighbor_indices)

            x = self.conv_norm(x)
            x = self.activation(x)

            if self.in_channels != self.out_channels:
                x = self.up_mlp(x)

        return x

    def __repr__(self):
        return "KPNextBlock(in_C: {:d}, out_C: {:d}, r: {:.2f}, mode: {:s})".format(
            self.in_channels,
            self.out_channels,
            self.radius,
            self.influence_mode,
        )


class KPNextResidualBlock(nn.Module):
    """
    Optional copied residual block.
    Kept for compatibility.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shell_sizes: list,
        radius: float,
        sigma: float,
        attention_groups: int = 8,
        attention_act: str = "sigmoid",
        mod_grp_norm: bool = False,
        shared_kp_data=None,
        influence_mode: str = "linear",
        dimension: int = 3,
        strided: bool = False,
        norm_type: str = "batch",
        bn_momentum: float = 0.1,
        activation: nn.Module = nn.LeakyReLU(0.1),
    ):
        super(KPNextResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strided = strided
        self.bn_momentum = bn_momentum
        self.norm_type = norm_type

        mid_channels = out_channels // 4

        if in_channels != mid_channels:
            self.unary1 = UnaryBlock(in_channels, mid_channels, norm_type, bn_momentum)
        else:
            self.unary1 = nn.Identity()

        if attention_groups == 0:
            self.conv = KPConvD(
                mid_channels,
                shell_sizes,
                radius,
                sigma,
                Cmid=0,
                shared_kp_data=shared_kp_data,
                dimension=dimension,
                influence_mode=influence_mode,
                norm_type=norm_type,
                bn_momentum=bn_momentum,
                activation=activation,
            )
        else:
            self.conv = DAKPConvX(
                mid_channels,
                shell_sizes,
                radius,
                sigma,
                attention_groups=attention_groups,
                attention_act=attention_act,
                mod_grp_norm=mod_grp_norm,
                shared_kp_data=shared_kp_data,
                dimension=dimension,
                influence_mode=influence_mode,
                norm_type=norm_type,
                bn_momentum=bn_momentum,
                activation=activation,
            )

        self.activation = activation
        self.norm = NormBlock(mid_channels, norm_type, bn_momentum)
        self.unary2 = UnaryBlock(mid_channels, out_channels, norm_type, bn_momentum, activation=None)

        if in_channels != out_channels:
            self.unary_shortcut = UnaryBlock(in_channels, out_channels, norm_type, bn_momentum, activation=None)
        else:
            self.unary_shortcut = nn.Identity()

    def forward(self, q_pts, s_pts, s_feats, neighbor_indices, da_scale=None):
        x = self.unary1(s_feats)

        if isinstance(self.conv, DAKPConvX):
            x = self.conv(q_pts, s_pts, x, neighbor_indices, da_scale=da_scale)
        else:
            x = self.conv(q_pts, s_pts, x, neighbor_indices)

        x = self.activation(self.norm(x))
        x = self.unary2(x)

        if self.strided:
            shortcut = local_maxpool(s_feats, neighbor_indices)
        else:
            shortcut = s_feats

        shortcut = self.unary_shortcut(shortcut)

        q_feats = x + shortcut
        q_feats = self.activation(q_feats)

        return q_feats


class KPNextInvertedBlock(nn.Module):
    """
    Optional copied inverted block.
    Kept for compatibility.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shell_sizes: list,
        radius: float,
        sigma: float,
        attention_groups: int = 8,
        attention_act: str = "sigmoid",
        mod_grp_norm: bool = False,
        expansion: int = 4,
        drop_path: float = -1.0,
        layer_scale_init_v: float = -1.0,
        shared_kp_data=None,
        influence_mode: str = "linear",
        dimension: int = 3,
        strided: bool = False,
        norm_type: str = "batch",
        bn_momentum: float = 0.1,
        activation: nn.Module = nn.LeakyReLU(0.1),
    ):
        super(KPNextInvertedBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.strided = strided
        self.bn_momentum = bn_momentum
        self.norm_type = norm_type
        mid_channels = out_channels * expansion

        if attention_groups == 0:
            self.conv = KPConvD(
                in_channels,
                shell_sizes,
                radius,
                sigma,
                Cmid=0,
                shared_kp_data=shared_kp_data,
                dimension=dimension,
                influence_mode=influence_mode,
                norm_type=norm_type,
                bn_momentum=bn_momentum,
                activation=activation,
            )
        else:
            self.conv = DAKPConvX(
                in_channels,
                shell_sizes,
                radius,
                sigma,
                attention_groups=attention_groups,
                attention_act=attention_act,
                mod_grp_norm=mod_grp_norm,
                shared_kp_data=shared_kp_data,
                dimension=dimension,
                influence_mode=influence_mode,
                norm_type=norm_type,
                bn_momentum=bn_momentum,
                activation=activation,
            )

        self.linear1 = nn.Linear(in_channels, mid_channels, bias=False)
        self.linear2 = nn.Linear(mid_channels, out_channels, bias=False)

        if in_channels != out_channels:
            self.linear_shortcut = nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.linear_shortcut = nn.Identity()

        self.norm0 = NormBlock(in_channels, norm_type, bn_momentum)
        self.norm1 = NormBlock(mid_channels, norm_type, bn_momentum)
        self.norm2 = NormBlock(out_channels, norm_type, bn_momentum)
        self.norm3 = NormBlock(out_channels, norm_type, bn_momentum)
        self.activation = activation

        if layer_scale_init_v > 0:
            self.gamma = nn.Parameter(layer_scale_init_v * torch.ones((out_channels)), requires_grad=True)
        else:
            self.gamma = None

        self.drop_path = DropPathPack(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, q_pts, s_pts, s_feats, neighbor_indices, da_scale=None):
        if isinstance(self.conv, DAKPConvX):
            x = self.conv(q_pts, s_pts, s_feats, neighbor_indices, da_scale=da_scale)
        else:
            x = self.conv(q_pts, s_pts, s_feats, neighbor_indices)

        x = self.norm0(x)
        x = self.activation(x)

        x = self.linear1(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.linear2(x)
        x = self.norm2(x)

        if self.gamma is not None:
            x = self.gamma * x

        if self.strided:
            shortcut = local_maxpool(s_feats, neighbor_indices)
        else:
            shortcut = s_feats

        shortcut = self.linear_shortcut(shortcut)
        shortcut = self.norm3(shortcut)

        q_feats = shortcut + self.drop_path(x)

        return q_feats


class DAKPNextMultiShortcutBlock(nn.Module):
    """
    DA-KPNext block used by KPConvXHybrid.

    This is the important block for your current model.

    Difference from baseline:
        forward(..., da_scale=None)
        self.conv = DAKPConvX(...)
        DAKPConvX internally scales kernel points using da_scale.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shell_sizes: list,
        radius: float,
        sigma: float,
        attention_groups: int = 8,
        attention_act: str = "sigmoid",
        mod_grp_norm: bool = False,
        expansion: int = 4,
        drop_path_p: float = -1.0,
        layer_scale_init_v: float = -1.0,
        use_upcut: bool = False,
        shared_kp_data=None,
        influence_mode: str = "linear",
        dimension: int = 3,
        norm_type: str = "batch",
        bn_momentum: float = 0.1,
        activation: nn.Module = nn.LeakyReLU(0.1),
    ):
        super(DAKPNextMultiShortcutBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.bn_momentum = bn_momentum
        self.norm_type = norm_type
        self.use_upcut = use_upcut
        mid_channels = in_channels * expansion

        if attention_groups == 0:
            self.conv = KPConvD(
                in_channels,
                shell_sizes,
                radius,
                sigma,
                Cmid=0,
                shared_kp_data=shared_kp_data,
                dimension=dimension,
                influence_mode=influence_mode,
                norm_type=norm_type,
                bn_momentum=bn_momentum,
                activation=activation,
            )
        else:
            self.conv = DAKPConvX(
                in_channels,
                shell_sizes,
                radius,
                sigma,
                attention_groups=attention_groups,
                attention_act=attention_act,
                mod_grp_norm=mod_grp_norm,
                shared_kp_data=shared_kp_data,
                dimension=dimension,
                influence_mode=influence_mode,
                norm_type=norm_type,
                bn_momentum=bn_momentum,
                activation=activation,
            )

        self.conv_norm = NormBlock(in_channels, norm_type, bn_momentum)

        self.activation = activation
        self.up_mlp = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            NormBlock(mid_channels, norm_type, bn_momentum),
        )
        self.down_mlp = nn.Sequential(
            nn.Linear(mid_channels, out_channels, bias=False),
            NormBlock(out_channels, norm_type, bn_momentum),
        )

        if in_channels != out_channels:
            self.mlp_downcut = nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                NormBlock(out_channels, norm_type, bn_momentum),
            )
        else:
            self.mlp_downcut = nn.Identity()

        if layer_scale_init_v > 0:
            self.gamma = nn.Parameter(layer_scale_init_v * torch.ones((out_channels)), requires_grad=True)
        else:
            self.gamma = None

        self.drop_path_p = drop_path_p
        if self.drop_path_p > 0:
            self.drop_path = DropPathPack(drop_path_p)

    def forward(
        self,
        q_pts,
        s_pts,
        s_feats,
        neighbor_indices,
        q_lengths,
        upcut=None,
        da_scale=None,
    ):
        downcut = s_feats

        if isinstance(self.conv, DAKPConvX):
            x = self.conv(
                q_pts,
                s_pts,
                s_feats,
                neighbor_indices,
                da_scale=da_scale,
            )
        else:
            x = self.conv(
                q_pts,
                s_pts,
                s_feats,
                neighbor_indices,
            )

        x = self.conv_norm(x)
        x = self.activation(x)

        x = self.up_mlp(x)

        if self.drop_path_p > 0:
            x, drop_mask = self.drop_path(x, q_lengths, return_mask=True)

        if upcut is not None:
            x = upcut + x

        x = self.activation(x)

        if self.use_upcut:
            upcut = x
        else:
            upcut = None

        x = self.down_mlp(x)

        if self.gamma is not None:
            x = self.gamma * x

        if self.drop_path_p > 0:
            x = drop_mask * x

        if self.in_channels != self.out_channels:
            downcut = self.mlp_downcut(downcut)

        x = downcut + x

        q_feats = self.activation(x)

        return q_feats, upcut
