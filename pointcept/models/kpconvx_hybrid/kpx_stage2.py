import math
import torch

from pointcept.models.builder import MODELS
from pointcept.models.kpconvx.utils.torch_pyramid import build_full_pyramid

from .kpx_stage1 import KPConvXStage1
from .da_kpconvx_block import DAKPXBlockAdapter


@MODELS.register_module()
class KPConvXStage2(KPConvXStage1):
    """
    Stage-2 improved KPConvX backbone.

    Stage-2 = Stage-1 + explicit Fine / Coarse dual-scale local adapter.

    Pipeline
    --------
    stem
    -> encoder stage
    -> dual-scale local adapter (selected stages)
    -> SGCA-lite (selected stages)
    -> pooling
    -> decoder
    -> head

    Notes
    -----
    - keep KPConvX local blocks unchanged
    - reuse existing neighbors from build_full_pyramid()
    - no dynamic graph rebuild
    - memory-aware implementation for 24 GB GPU training
    """

    def __init__(
        self,
        in_channels=None,
        input_channels=None,
        num_classes=13,
        enable_da=True,
        da_stages=(2, 3, 4),
        da_dropout=0.0,
        da_hidden_ratio=0.75,
        da_scale_range=(0.80, 1.25),
        da_fine_scale=0.85,
        da_coarse_scale=1.20,
        da_use_density=True,
        init_channels=64,
        channel_scaling=math.sqrt(2),
        **kwargs
    ):
        if input_channels is None:
            input_channels = in_channels
        if input_channels is None:
            raise ValueError("Either `in_channels` or `input_channels` must be provided.")

        self._stage2_init_channels = init_channels
        self._stage2_channel_scaling = channel_scaling

        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            init_channels=init_channels,
            channel_scaling=channel_scaling,
            **kwargs
        )

        self.enable_da = enable_da
        self.da_stages = tuple(da_stages) if enable_da else tuple()

        layer_channels = self._compute_layer_channels(
            init_channels=self._stage2_init_channels,
            channel_scaling=self._stage2_channel_scaling,
            num_layers=self.num_layers,
        )

        for stage in self.da_stages:
            if stage < 1 or stage > self.num_layers:
                raise ValueError(f"Invalid DA stage index: {stage}")

            # When grid_pool is enabled, the last block of each stage outputs
            # the next layer's channels. So DA block should use the next layer's dim.
            if self.grid_pool and stage < self.num_layers:
                dim = layer_channels[stage]
            else:
                dim = layer_channels[stage - 1]
            setattr(
                self,
                f"da_{stage}",
                DAKPXBlockAdapter(
                    dim=dim,
                    hidden_ratio=da_hidden_ratio,
                    dropout=da_dropout,
                    scale_range=da_scale_range,
                    fine_scale=da_fine_scale,
                    coarse_scale=da_coarse_scale,
                    use_density=da_use_density,
                ),
            )

    def _apply_da_if_needed(self, stage_idx, feats, points, neighbors):
        if not self.enable_da:
            return feats
        if stage_idx not in self.da_stages:
            return feats
        da_block = getattr(self, f"da_{stage_idx}")
        return da_block(feats, points, neighbors)

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
                upcut = None
                for block in block_list:
                    feats, upcut = block(
                        in_dict.points[l],
                        in_dict.points[l],
                        feats,
                        in_dict.neighbors[l],
                        in_dict.lengths[l],
                        upcut=upcut,
                    )

            # Fine / Coarse dual-scale local adapter
            feats = self._apply_da_if_needed(
                stage_idx=layer,
                feats=feats,
                points=in_dict.points[l],
                neighbors=in_dict.neighbors[l],
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
