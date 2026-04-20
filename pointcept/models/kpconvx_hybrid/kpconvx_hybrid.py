import math
import torch

from pointcept.models.builder import MODELS
from pointcept.models.kpconvx.utils.torch_pyramid import build_full_pyramid

from .decoder_refine import DecoderRefineHead
from .geometry_router import GeometryDifficultyRouter
from .kpx_stage2 import KPConvXStage2


@MODELS.register_module("KPConvXHybrid")
class KPConvXHybrid(KPConvXStage2):
    """
    Final hybrid backbone used in this project.

    Effective pipeline:
        KPConvX base backbone
        -> geometry difficulty router
        -> Stage-2 dynamic local adapter (DA)
        -> SGCA-lite global context
        -> difficulty-aware global fusion
        -> decoder
        -> lightweight decoder refine
        -> segmentation head

    Notes:
        - keep KPConvXBase as the true backbone主体
        - do not rebuild dynamic graph
        - router changes actual model behavior
        - preserve Pointcept DefaultSegmentor compatibility
    """

    def __init__(
        self,
        in_channels=None,
        input_channels=None,
        num_classes=13,
        enable_router=True,
        router_cfg=None,
        global_cfg=None,
        fusion_cfg=None,
        refine_cfg=None,
        router_stages=(2, 3, 4),
        router_hidden_ratio=0.5,
        router_dropout=0.0,
        router_temperature=1.0,
        router_local_boost=1.0,
        router_global_boost=1.0,
        enable_global=True,
        global_stages=(4, 5),
        global_patch_sizes=(192, 320),
        global_num_heads=(8, 16),
        global_mlp_ratio=2.0,
        global_dropout=0.0,
        enable_da=True,
        da_stages=(2, 3, 4),
        da_dropout=0.0,
        da_scale_range=(0.75, 1.35),
        da_branch_scales=(0.85, 1.25),
        enable_refine=True,
        refine_hidden_ratio=0.5,
        refine_dropout=0.0,
        refine_use_coords=True,
        refine_use_boundary=True,
        init_channels=64,
        channel_scaling=math.sqrt(2),
        **kwargs,
    ):
        if input_channels is None:
            input_channels = in_channels
        if input_channels is None:
            raise ValueError("Either `in_channels` or `input_channels` must be provided.")

        self.router_cfg = router_cfg or {}
        self.global_cfg = global_cfg or {}
        self.fusion_cfg = fusion_cfg or {}
        self.refine_cfg = refine_cfg or {}

        enable_router = self.router_cfg.get("enable", enable_router)
        router_stages = self.router_cfg.get("stages", router_stages)
        router_hidden_ratio = self.router_cfg.get("hidden_ratio", router_hidden_ratio)
        router_dropout = self.router_cfg.get("dropout", router_dropout)
        router_temperature = self.router_cfg.get("temperature", router_temperature)
        router_local_boost = self.router_cfg.get("local_boost", router_local_boost)
        router_global_boost = self.router_cfg.get("global_boost", router_global_boost)

        refine_hidden_ratio = self.refine_cfg.get("hidden_ratio", refine_hidden_ratio)
        refine_dropout = self.refine_cfg.get("dropout", refine_dropout)
        refine_use_coords = self.refine_cfg.get("use_coords", refine_use_coords)
        refine_use_boundary = self.refine_cfg.get("use_boundary", refine_use_boundary)

        self.enable_router = bool(enable_router)
        self.router_stages = tuple(router_stages) if self.enable_router else tuple()

        self._hybrid_init_channels = init_channels
        self._hybrid_channel_scaling = channel_scaling

        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            enable_global=enable_global,
            global_stages=global_stages,
            global_patch_sizes=global_patch_sizes,
            global_num_heads=global_num_heads,
            global_mlp_ratio=global_mlp_ratio,
            global_dropout=global_dropout,
            enable_da=enable_da,
            da_stages=da_stages,
            da_dropout=da_dropout,
            da_scale_range=da_scale_range,
            da_branch_scales=da_branch_scales,
            init_channels=init_channels,
            channel_scaling=channel_scaling,
            **kwargs,
        )

        layer_channels = self._compute_layer_channels(
            init_channels=self._hybrid_init_channels,
            channel_scaling=self._hybrid_channel_scaling,
            num_layers=self.num_layers,
        )

        if self.enable_router:
            for stage in self.router_stages:
                if stage < 1 or stage > self.num_layers:
                    raise ValueError(f"Invalid router stage index: {stage}")
                # When grid_pool is enabled, the last block of each stage outputs
                # the next layer's channels. So router should use the next layer's dim.
                if self.grid_pool and stage < self.num_layers:
                    dim = layer_channels[stage]
                else:
                    dim = layer_channels[stage - 1]
                hidden_dim = max(32, int(dim * float(router_hidden_ratio)))
                setattr(
                    self,
                    f"router_{stage}",
                    GeometryDifficultyRouter(
                        dim=dim,
                        hidden_dim=hidden_dim,
                        dropout=router_dropout,
                        temperature=router_temperature,
                        local_boost=router_local_boost,
                        global_boost=router_global_boost,
                    ),
                )

        self.enable_refine = bool(enable_refine) and self.task == "cloud_segmentation"
        if self.enable_refine:
            refine_dim = layer_channels[0]
            refine_hidden_dim = max(32, int(refine_dim * float(refine_hidden_ratio)))
            self.decoder_refine = DecoderRefineHead(
                dim=refine_dim,
                hidden_dim=refine_hidden_dim,
                dropout=refine_dropout,
                use_coords=refine_use_coords,
                use_boundary=refine_use_boundary,
            )

    def _route_stage_if_needed(self, stage_idx, feats, points, neighbors):
        if not self.enable_router:
            return None
        if stage_idx not in self.router_stages:
            return None
        router = getattr(self, f"router_{stage_idx}")
        return router(feats, points, neighbors)

    def _apply_da_if_needed(self, stage_idx, feats, points, neighbors, router_state=None):
        if not self.enable_da:
            return feats
        if stage_idx not in self.da_stages:
            return feats
        da_block = getattr(self, f"da_{stage_idx}")
        route_logits = None if router_state is None else router_state.get("local_logits")
        return da_block(feats, points, neighbors, route_logits=route_logits)

    def _apply_global_with_router(self, stage_idx, feats, points, lengths, router_state=None):
        out = self._apply_sgca_if_needed(
            stage_idx=stage_idx,
            feats=feats,
            points=points,
            lengths=lengths,
        )
        if router_state is None:
            return out

        global_weight = router_state.get("global_weight")
        if global_weight is None:
            return out
        return feats + global_weight * (out - feats)

    def _apply_refine_if_needed(self, feats, points, neighbors):
        if not self.enable_refine:
            return feats
        return self.decoder_refine(feats, points, neighbors)

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

            router_state = self._route_stage_if_needed(
                stage_idx=layer,
                feats=feats,
                points=in_dict.points[l],
                neighbors=in_dict.neighbors[l],
            )

            feats = self._apply_da_if_needed(
                stage_idx=layer,
                feats=feats,
                points=in_dict.points[l],
                neighbors=in_dict.neighbors[l],
                router_state=router_state,
            )

            feats = self._apply_global_with_router(
                stage_idx=layer,
                feats=feats,
                points=in_dict.points[l],
                lengths=in_dict.lengths[l],
                router_state=router_state,
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

            feats = self._apply_refine_if_needed(
                feats=feats,
                points=in_dict.points[0],
                neighbors=in_dict.neighbors[0],
            )

        # ------ Head ------
        logits = self.head(feats)
        return logits
