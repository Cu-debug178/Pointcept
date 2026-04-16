"""
Unified hybrid KPConvX backbone.

Current composition:
    KPConvXBase main body
    + Stage-1 SGCA global context
    + Stage-2 DA fine/coarse local adapter
    + decoder refine module

This file intentionally provides one stable registry entry:
    type="KPConvXHybrid"
"""

import math
import torch

from pointcept.models.builder import MODELS
from pointcept.models.kpconvx.utils.torch_pyramid import build_full_pyramid

from .kpx_stage2 import KPConvXStage2
from .decoder_refine import DecoderRefineModule


@MODELS.register_module("KPConvXHybrid")
class KPConvXHybrid(KPConvXStage2):
    def __init__(
        self,
        in_channels=None,
        input_channels=None,
        num_classes=13,
        enable_refine=True,
        refine_dropout=0.0,
        refine_hidden_ratio=1.0,
        refine_use_coord=True,
        init_channels=64,
        channel_scaling=math.sqrt(2),
        **kwargs,
    ):
        if input_channels is None:
            input_channels = in_channels
        if input_channels is None:
            raise ValueError("Either `in_channels` or `input_channels` must be provided.")

        self.enable_refine = enable_refine
        self.refine_dropout = refine_dropout
        self.refine_hidden_ratio = refine_hidden_ratio
        self.refine_use_coord = refine_use_coord
        self._hybrid_init_channels = init_channels
        self._hybrid_channel_scaling = channel_scaling

        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            init_channels=init_channels,
            channel_scaling=channel_scaling,
            **kwargs,
        )

        if self.task != "cloud_segmentation":
            self.enable_refine = False

        if self.enable_refine:
            finest_dim = self._compute_layer_channels(
                init_channels=self._hybrid_init_channels,
                channel_scaling=self._hybrid_channel_scaling,
                num_layers=self.num_layers,
            )[0]
            hidden_dim = max(finest_dim, int(round(finest_dim * self.refine_hidden_ratio)))
            self.decoder_refine = DecoderRefineModule(
                dim=finest_dim,
                hidden_dim=hidden_dim,
                dropout=refine_dropout,
                use_coord=refine_use_coord,
            )
        else:
            self.decoder_refine = None

    def _apply_refine_if_needed(self, feats, points, neighbors):
        if not self.enable_refine or self.decoder_refine is None:
            return feats
        return self.decoder_refine(feats, points, neighbors)

    def forward(self, data_dict):
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

        feats = self.stem(
            in_dict.points[0],
            in_dict.points[0],
            feats,
            in_dict.neighbors[0],
        )

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

            feats = self._apply_da_if_needed(
                stage_idx=layer,
                feats=feats,
                points=in_dict.points[l],
                neighbors=in_dict.neighbors[l],
            )

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

            feats = self._apply_refine_if_needed(
                feats=feats,
                points=in_dict.points[0],
                neighbors=in_dict.neighbors[0],
            )

        logits = self.head(feats)
        return logits
