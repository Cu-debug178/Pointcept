_base_ = ["../_base_/default_runtime.py"]

# runtime (single 4090D 24G friendly)
num_worker = 8

# 修改作用：
# 加入 DA-KPConvX + SGCA + refine 后显存会更高。
# 如果 batch_size=3 报 OOM，先改成 batch_size=2。
batch_size = 3

# batch_size_test = 8 设置8报错了
mix_prob = 0.8
max_input_pts = 40000

# 梯度裁剪
clip_grad = 1.0

# 修改建议：
# 如果环境 AMP 稳定，建议改成 True，可以明显降低显存。
# 第一次如果想先排查代码错误，也可以保持 False。
enable_amp = False

# model
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="KPConvXHybrid",
        input_channels=9,
        num_classes=13,
        dim=3,
        task="cloud_segmentation",

        # ------------------------------------------------------------
        # KPConvX baseline backbone settings
        # ------------------------------------------------------------
        kp_mode="kpconvx",
        shell_sizes=(1, 14, 28),
        kp_radius=2.3,
        kp_aggregation="nearest",

        # 修改说明：
        # 这里先保留 constant，保证第一次能跑通。
        # DA-KPConvX 在 constant 下仍然会影响 nearest kernel assignment。
        # 如果后续想让 DA 缩放对距离权重影响更明显，可以再尝试 kp_influence="linear"。
        kp_influence="constant",

        kp_sigma=2.3,
        share_kp=False,
        conv_groups=-1,
        inv_groups=8,
        inv_act="sigmoid",
        inv_grp_norm=True,
        kpx_upcut=False,

        # ------------------------------------------------------------
        # Pyramid / encoder settings
        # ------------------------------------------------------------
        subsample_size=0.02,
        neighbor_limits=(12, 16, 20, 20, 20),
        layer_blocks=(3, 3, 9, 12, 3),
        init_channels=64,
        channel_scaling=1.414,
        radius_scaling=2.2,
        decoder_layer=True,
        grid_pool=True,
        upsample_n=3,
        first_inv_layer=1,
        drop_path_rate=0.3,
        norm="batch",
        bn_momentum=0.1,
        smooth_labels=False,
        class_w=(),

        # ------------------------------------------------------------
        # Router: only controls SGCA global fusion
        # ------------------------------------------------------------
        # 修改作用：
        # 旧版本 router 会输出 local_logits，用来控制 Fine / Coarse DA adapter。
        # 现在 DA 已经改成 kernel point scaling，不再有 Fine / Coarse adapter。
        # 所以 router 只保留 global_weight，用来控制 SGCA residual 注入强度。
        enable_router=True,
        router_stages=(2, 3, 4),
        router_hidden_ratio=0.5,
        router_dropout=0.0,
        router_temperature=1.0,

        # 修改作用：
        # 保留 global_boost，用于增强 difficulty 对 SGCA global residual 的影响。
        router_global_boost=1.0,

        # 已删除：
        # router_local_boost=1.0,
        # 删除原因：
        # 新版 geometry_router.py 不再输出 local_logits，
        # DA-KPConvX 不再使用 router 控制 small/large branch。

        # ------------------------------------------------------------
        # Stage-1 global context
        # ------------------------------------------------------------
        # 作用：
        # SGCA 全局上下文增强模块。
        # 仍然保留，用于补充长程依赖和全局语义信息。
        enable_global=True,
        global_stages=(4, 5),
        global_patch_sizes=(192, 320),
        global_num_heads=(8, 16),
        global_mlp_ratio=2.0,
        global_dropout=0.0,

        # ------------------------------------------------------------
        # DA-KPConvX: density-adaptive kernel scaling
        # ------------------------------------------------------------
        # 修改作用：
        # 旧版本是 Stage-2 density-aware local adapter：
        #   KPConvX 后面额外接 Fine / Coarse 双分支特征 adapter。
        #
        # 新版本改成真正的 DA-KPConvX kernel scaling：
        #   1. 计算局部密度 rho_i
        #   2. 映射为缩放因子 s_i
        #   3. 在 DAKPConvX 内部缩放 kernel points:
        #          p_k_da = s_i * p_k
        #
        # 这样 DA 直接进入 KPConvX 的 kernel influence 计算，而不是后处理特征。
        enable_da=True,

        # 修改作用：
        # 在第 2、3、4 个 encoder stage 使用 DA kernel scaling。
        # 第 1 层点太密、低级几何较敏感；第 5 层点太少、全局语义更强。
        # 所以先用 (2, 3, 4) 比较稳。
        da_stages=(2, 3, 4),

        # 修改作用：
        # 方案 A：归一化线性映射。
        # 密集区 rho 高 -> s_i 接近 0.5，kernel 感受野收缩。
        # 稀疏区 rho 低 -> s_i 接近 2.0，kernel 感受野扩张。
        #
        # 旧配置是 da_scale_range=(0.75, 1.35)，动态范围偏保守。
        # 现在按你的公式推荐改成 (0.5, 2.0)。
        da_scale_range=(0.5, 2.0),

        # 修改作用：
        # 用前 16 个邻居估计局部密度：
        #   rho_i = 1 / (mean_neighbor_distance_i + eps)
        #
        # 当前 neighbor_limits=(12, 16, 20, 20, 20)，
        # DA stages=(2, 3, 4)，所以：
        #   stage2 使用最多 16 个邻居
        #   stage3 使用前 16 个邻居
        #   stage4 使用前 16 个邻居
        da_density_k=16,

        # 已删除：
        # da_dropout=0.0,
        # da_branch_scales=(0.85, 1.25),
        #
        # 删除原因：
        # 这两个参数属于旧的 Fine / Coarse feature adapter。
        # 新版 DA-KPConvX 不再使用 DAKPXBlockAdapter，
        # 因此不需要 da_dropout / da_branch_scales。

        # ------------------------------------------------------------
        # Hybrid refine
        # ------------------------------------------------------------
        # 作用：
        # decoder 后的轻量细节恢复模块，用于边界和局部细节 refinement。
        enable_refine=True,
        refine_hidden_ratio=0.5,
        refine_dropout=0.0,
        refine_use_coords=True,
        refine_use_boundary=True,
    ),
    criteria=[
        dict(
            type="CrossEntropyLoss",
            loss_weight=1.0,
            ignore_index=-1,
        ),
        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=1.0,
            ignore_index=-1,
        ),
    ],
)

# scheduler
epoch = 1000
eval_epoch = 200
optimizer = dict(type="AdamW", lr=0.005, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=100.0,
    final_div_factor=1000.0,
)

# dataset
dataset_type = "S3DISDataset"
data_root = "/root/autodl-tmp/data/s3dis"

data = dict(
    num_classes=13,
    ignore_index=-1,
    names=[
        "ceiling",
        "floor",
        "wall",
        "beam",
        "column",
        "window",
        "door",
        "table",
        "chair",
        "sofa",
        "bookcase",
        "board",
        "clutter",
    ],
    train=dict(
        type=dataset_type,
        split=["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"],
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(
                type="RandomRotateTargetAngle",
                angle=(1 / 2, 1, 3 / 2),
                center=[0, 0, 0],
                axis="z",
                p=0.75,
            ),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.0),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_min_coord=True,
            ),
            dict(type="SphereCrop", point_max=max_input_pts, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="Area_5",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_min_coord=True,
            ),
            dict(type="SphereCrop", point_max=max_input_pts, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="Area_5",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "index"),
                    feat_keys=("coord", "color", "normal"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)
