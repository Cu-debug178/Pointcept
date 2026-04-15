import math
import torch

from pointcept.models.builder import MODELS
from pointcept.models.kpconvx.kpconvx_base import KPConvXBase
from pointcept.models.kpconvx.utils.torch_pyramid import build_full_pyramid

from .sgca import SGCALite


@MODELS.register_module()
class KPConvXStage1(KPConvXBase):
    """
    Stage-1 improved KPConvX backbone.

    Main idea:
        - keep KPConvX as the main local geometry backbone
        - insert SGCA-lite only at low-resolution encoder stages
        - keep DefaultSegmentor compatibility (backbone returns logits)

    Stage mapping for default 5-layer KPConvX:
        stage 1 -> high resolution
        stage 2 -> high resolution
        stage 3 -> low resolution
        stage 4 -> lower resolution
        stage 5 -> bottleneck
    """

    def __init__(
        self,
        in_channels=None,
        input_channels=None,
        num_classes=13,
        enable_global=True,
        global_stages=(3, 4, 5),
        global_patch_sizes=(128, 256, 384),
        global_num_heads=(8, 8, 16),
        global_mlp_ratio=2.0,
        global_dropout=0.0,
        init_channels=64,
        channel_scaling=math.sqrt(2),
        **kwargs
    ):
        if input_channels is None:
            input_channels = in_channels
        if input_channels is None:
            raise ValueError("Either `in_channels` or `input_channels` must be provided.")

        self._stage1_init_channels = init_channels
        self._stage1_channel_scaling = channel_scaling

        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            init_channels=init_channels,
            channel_scaling=channel_scaling,
            **kwargs
        )

        self.enable_global = enable_global
        self.global_stages = tuple(global_stages) if enable_global else tuple()

        layer_channels = self._compute_layer_channels(
            init_channels=self._stage1_init_channels,
            channel_scaling=self._stage1_channel_scaling,
            num_layers=self.num_layers,
        )

        for i, stage in enumerate(self.global_stages):
            if stage < 1 or stage > self.num_layers:
                raise ValueError(f"Invalid global stage index: {stage}")

            dim = layer_channels[stage - 1]
            patch_size = global_patch_sizes[min(i, len(global_patch_sizes) - 1)]
            num_heads = self._safe_num_heads(
                dim=dim,
                requested_heads=global_num_heads[min(i, len(global_num_heads) - 1)],
            )

            setattr(
                self,
                f"sgca_{stage}",
                SGCALite(
                    dim=dim,
                    num_heads=num_heads,
                    patch_size=patch_size,
                    mlp_ratio=global_mlp_ratio,
                    dropout=global_dropout,
                ),
            )

    @staticmethod
    def _compute_layer_channels(init_channels, channel_scaling, num_layers):
        layer_channels = []
        for l in range(num_layers):
            target_c = init_channels * (channel_scaling ** l)
            layer_channels.append(int(math.ceil((target_c - 0.1) / 16)) * 16)
        return layer_channels

    @staticmethod
    def _safe_num_heads(dim, requested_heads):
        heads = max(1, min(dim, int(requested_heads)))
        while dim % heads != 0 and heads > 1:
            heads -= 1
        return heads

    def _apply_sgca_if_needed(self, stage_idx, feats, points, lengths):
        if not self.enable_global:
            return feats
        if stage_idx not in self.global_stages:
            return feats
        sgca = getattr(self, f"sgca_{stage_idx}")
        return sgca(feats, points, lengths)

    def forward(self, data_dict):
        # ------ Init ------
        points = data_dict["coord"]
        feats = data_dict["feat"]
        offset = data_dict["offset"].int()

        # Convert offsets to lengths
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

            # Stage-1 global context insertion
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
