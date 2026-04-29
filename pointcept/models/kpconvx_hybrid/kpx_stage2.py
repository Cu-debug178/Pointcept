import math
import torch

from pointcept.models.builder import MODELS
from pointcept.models.kpconvx.utils.torch_pyramid import build_full_pyramid

from .kpx_stage1 import KPConvXStage1
from .da_kpconvx_block import DensityAdaptiveScale
from .da_kpnext_blocks import DAKPNextMultiShortcutBlock


@MODELS.register_module()
class KPConvXStage2(KPConvXStage1):
    """
    Stage-2 improved KPConvX backbone.

    Stage-2 = Stage-1 + Density-Adaptive KPConvX.

    Main change
    -----------
    Instead of adding a feature-level Fine / Coarse adapter after KPConvX,
    this version computes a density-aware scale s_i for each query point and
    passes it into DA-KPConvX blocks.

    DA-KPConvX kernel scaling
    -------------------------
        rho_i = 1 / (mean_neighbor_distance_i + eps)

        dense region:
            rho_i high -> s_i small

        sparse region:
            rho_i low -> s_i large

        kernel point scaling:
            p_k_da = s_i * p_k

    Pipeline
    --------
    stem
    -> encoder stage with DA-KPConvX kernel scaling
    -> SGCA global context branch from Stage-1
    -> pooling
    -> decoder
    -> head

    Notes
    -----
    - Original baseline KPConvX files are not modified.
    - This class uses DAKPNextMultiShortcutBlock from kpconvx_hybrid.
    - No dynamic graph rebuild.
    - Existing pyramid neighbors from build_full_pyramid() are reused.
    """

    def __init__(
        self,
        in_channels=None,
        input_channels=None,
        num_classes=13,
        enable_da=True,
        da_stages=(2, 3, 4),
        da_scale_range=(0.5, 2.0),
        da_density_k=16,
        init_channels=64,
        channel_scaling=math.sqrt(2),
        **kwargs,
    ):
        if input_channels is None:
            input_channels = in_channels

        if input_channels is None:
            raise ValueError("Either `in_channels` or `input_channels` must be provided.")

        self._stage2_init_channels = init_channels
        self._stage2_channel_scaling = channel_scaling

        # DA settings must be assigned before super().__init__().
        # KPConvXStage1 / KPConvXBase may call self.get_residual_block()
        # during initialization, so the DA block factory must already be ready.
        self.enable_da = bool(enable_da)
        self.da_stages = tuple(da_stages) if enable_da else tuple()
        self.da_scale_range = da_scale_range
        self.da_density_k = da_density_k

        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            init_channels=init_channels,
            channel_scaling=channel_scaling,
            **kwargs,
        )

        for stage in self.da_stages:
            if stage < 1 or stage > self.num_layers:
                raise ValueError(f"Invalid DA stage index: {stage}")

        self.da_scale = DensityAdaptiveScale(
            scale_range=da_scale_range,
            density_k=da_density_k,
        )

    def get_residual_block(
        self,
        in_C,
        out_C,
        radius,
        sigma,
        shared_kp_data=None,
        conv_layer=False,
        drop_path=-1,
    ):
        """
        Build encoder residual blocks.

        This overrides the block factory from KPConvXStage1 / KPConvXBase so
        that the improved model uses DAKPNextMultiShortcutBlock, while the
        original KPConvX baseline remains untouched.

        For kpconv / kpconvtest mode, fall back to the original implementation.
        For KPConvX mode, use the DA-compatible block.
        """

        if self.kp_mode in ["kpconv", "kpconvtest"]:
            return super().get_residual_block(
                in_C,
                out_C,
                radius,
                sigma,
                shared_kp_data=shared_kp_data,
                conv_layer=conv_layer,
                drop_path=drop_path,
            )

        attention_groups = self.inv_groups

        if conv_layer or "kpconvd" in self.kp_mode:
            attention_groups = 0

        return DAKPNextMultiShortcutBlock(
            in_C,
            out_C,
            self.shell_sizes,
            radius,
            sigma,
            attention_groups=attention_groups,
            attention_act=self.inv_act,
            mod_grp_norm=self.inv_grp_norm,
            expansion=4,
            drop_path_p=drop_path,
            layer_scale_init_v=-1.0,
            use_upcut=self.kpx_upcut,
            shared_kp_data=shared_kp_data,
            influence_mode=self.kp_influence,
            dimension=self.dim,
            norm_type=self.norm,
            bn_momentum=self.bn_momentum,
        )

    def _get_da_scale_if_needed(self, stage_idx, points, neighbors, lengths):
        """
        Compute density-adaptive scale for selected encoder stages.

        Return:
            None if DA is disabled or this stage is not selected.
            Tensor [N, 1] otherwise.
        """

        if not self.enable_da:
            return None

        if stage_idx not in self.da_stages:
            return None

        return self.da_scale(
            points=points,
            neighbors=neighbors,
            lengths=lengths,
        )

    def forward(self, data_dict):
        # ------ Init ------
        points = data_dict["coord"]
        feats = data_dict["feat"]
        offset = data_dict["offset"].int()

        offset = torch.cat(
            [torch.zeros(1, dtype=offset.dtype, device=offset.device), offset],
            dim=0,
        )
        lengths = offset[1:] - offset[:-1]

        in_dict = build_full_pyramid(
            points,
            lengths,
            self.num_layers,
            self.subsample_size,
            self.first_radius,
            self.radius_scaling,
            self.neighbor_limits,
            self.upsample_n,
            sub_mode=self.in_sub_mode,
            grid_pool_mode=self.grid_pool,
        )

        # ------ Stem ------
        feats = self.stem(
            in_dict.points[0],
            in_dict.points[0],
            feats,
            in_dict.neighbors[0],
        )

        # ------ Encoder ------
        skip_feats = []

        for layer in range(1, self.num_layers + 1):
            l = layer - 1
            block_list = getattr(self, f"encoder_{layer}")

            if self.kp_mode in ["kpconv", "kpconvtest"]:
                for block in block_list:
                    feats = block(
                        in_dict.points[l],
                        in_dict.points[l],
                        feats,
                        in_dict.neighbors[l],
                    )

            else:
                da_scale = self._get_da_scale_if_needed(
                    stage_idx=layer,
                    points=in_dict.points[l],
                    neighbors=in_dict.neighbors[l],
                    lengths=in_dict.lengths[l],
                )

                upcut = None
                for block in block_list:
                    feats, upcut = block(
                        in_dict.points[l],
                        in_dict.points[l],
                        feats,
                        in_dict.neighbors[l],
                        in_dict.lengths[l],
                        upcut=upcut,
                        da_scale=da_scale,
                    )

            # Global context branch from Stage-1
            feats = self._apply_sgca_if_needed(
                stage_idx=layer,
                feats=feats,
                points=in_dict.points[l],
                lengths=in_dict.lengths[l],
            )

            if layer < self.num_layers:
                skip_feats.append(feats)

                layer_pool = getattr(self, f"pooling_{layer}")

                if self.grid_pool:
                    if isinstance(in_dict.pools[l], tuple):
                        feats = layer_pool(
                            feats,
                            in_dict.pools[l][0],
                            idx_ptr=in_dict.pools[l][1],
                        )
                    else:
                        feats = layer_pool(feats, in_dict.pools[l])

                else:
                    feats = layer_pool(
                        in_dict.points[l + 1],
                        in_dict.points[l],
                        feats,
                        in_dict.pools[l],
                    )

        # ------ Decoder / Classification ------
        if self.task == "classification":
            feats = self.global_pooling(feats, in_dict.lengths[-1])

        elif self.task == "cloud_segmentation":
            for layer in range(self.num_layers - 1, 0, -1):
                l = layer - 1
                upsample = getattr(self, f"upsampling_{layer}")

                if self.grid_pool:
                    feats = upsample(feats, in_dict.upsamples[l])
                else:
                    feats = upsample(
                        feats,
                        in_dict.upsamples[l],
                        in_dict.up_distances[l],
                    )

                feats = torch.cat([feats, skip_feats[l]], dim=1)

                unary = getattr(self, f"decoder_unary_{layer}")
                feats = unary(feats)

                if self.add_decoder_layer:
                    block = getattr(self, f"decoder_layer_{layer}")

                    if self.kp_mode in ["kpconv", "kpconvtest"]:
                        feats = block(
                            in_dict.points[l],
                            in_dict.points[l],
                            feats,
                            in_dict.neighbors[l],
                        )

                    else:
                        # Decoder remains unchanged for stability.
                        feats, _ = block(
                            in_dict.points[l],
                            in_dict.points[l],
                            feats,
                            in_dict.neighbors[l],
                            in_dict.lengths[l],
                        )

        # ------ Head ------
        logits = self.head(feats)
        return logits
