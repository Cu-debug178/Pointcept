"""
KPConvXHybrid: minimal runnable hybrid model for S3DIS in Pointcept.

Stage-0 target:
    - Keep the model directly runnable in Pointcept.
    - Reuse KPConvXBase as the actual working backbone.
    - Reserve interfaces for future modules:
        1) Geometry Difficulty Router
        2) Sparse Global Context Branch
        3) Feature Fusion
        4) Boundary-aware Refinement Head

Planned full pipeline:
    Input
    -> KPConvX stem
    -> Encoder stage 1
    -> Encoder stage 2
    -> Geometry Difficulty Router
    -> Sparse Global Context Branch
    -> Feature Fusion
    -> Encoder stage 3
    -> Encoder stage 4
    -> Decoder
    -> Boundary-aware Refinement Head
    -> Point-wise Segmentation

Current stage:
    This file only provides a minimal wrapper over KPConvXBase.
    Hybrid modules are not activated yet.
"""

from pointcept.models.builder import MODELS
from pointcept.models.kpconvx.kpconvx_base import KPConvXBase


@MODELS.register_module("KPConvXHybrid")
class KPConvXHybrid(KPConvXBase):
    """
    Minimal runnable hybrid wrapper.

    Notes
    -----
    Stage-0 behavior:
        output = KPConvXBase(input)

    Future hybrid idea:
        Let F_2 denote the feature after encoder stage 2.
        A difficulty router will estimate:
            s_i = Router(F_2, geometry_i)
        where s_i is the difficulty score for region/point i.

        Then a sparse global branch will process hard regions:
            G = GlobalContext(F_2, s)

        Fusion will combine local and global features:
            F_2^fused = Fusion(F_2, G)

        Finally decoder and refinement head will predict segmentation logits.
    """

    def __init__(
        self,
        in_channels=6,
        num_classes=13,
        enable_router=False,
        enable_global=False,
        enable_refine=False,
        router_cfg=None,
        global_cfg=None,
        fusion_cfg=None,
        refine_cfg=None,
        **kwargs
    ):
        # Save future switches for later development.
        self.enable_router = enable_router
        self.enable_global = enable_global
        self.enable_refine = enable_refine

        self.router_cfg = router_cfg or {}
        self.global_cfg = global_cfg or {}
        self.fusion_cfg = fusion_cfg or {}
        self.refine_cfg = refine_cfg or {}

        # IMPORTANT:
        # KPConvXBase expects `input_channels`, not `in_channels`.
        super().__init__(
            input_channels=in_channels,
            num_classes=num_classes,
            **kwargs
        )

    def forward(self, data_dict):
        """
        Stage-0 forward.

        Current implementation:
            logits = KPConvXBase(data_dict)

        Future implementation sketch:
            1) Extract stage-2 features F_2
            2) Compute difficulty score:
                   s = Router(F_2, geometry)
            3) Select or weight hard regions
            4) Compute sparse global context:
                   G = GlobalContext(F_2, s)
            5) Fuse features:
                   F_2^fused = Fusion(F_2, G)
            6) Continue encoder / decoder / refinement
            7) Output segmentation logits
        """
        return super().forward(data_dict)


from pointcept.models.builder import MODELS
from .kpx_stage1 import KPConvXStage1


@MODELS.register_module("KPConvXHybrid")
class KPConvXHybrid(KPConvXStage1):
    """
    Backward-compatible alias.

    Old experiments that still use:
        type="KPConvXHybrid"

    will automatically use the Stage-1 implementation.
    """
    pass
