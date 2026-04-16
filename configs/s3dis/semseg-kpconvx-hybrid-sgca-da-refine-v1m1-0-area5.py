_base_ = ["../_base_/default_runtime.py"]

# runtime (single 4090D 24G friendly)
num_worker = 8
batch_size = 1
mix_prob = 0.8
enable_amp = True
max_input_pts = 30000

# model
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="KPConvXHybrid",
        input_channels=9,
        num_classes=13,
        dim=3,
        task="cloud_segmentation",
        kp_mode="kpconvx",
        shell_sizes=(1, 14, 28),
        kp_radius=2.3,
        kp_aggregation="nearest",
        kp_influence="constant",
        kp_sigma=2.3,
        share_kp=False,
        conv_groups=-1,
        inv_groups=8,
        inv_act="sigmoid",
        inv_grp_norm=True,
        kpx_upcut=False,
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

        # Stage-1 global context
        enable_global=True,
        global_stages=(4, 5),
        global_patch_sizes=(192, 320),
        global_num_heads=(8, 16),
        global_mlp_ratio=2.0,
        global_dropout=0.0,

        # Stage-2 density-aware local adapter
        enable_da=True,
        da_stages=(2, 3, 4),
        da_dropout=0.0,
        da_scale_range=(0.75, 1.35),
        da_branch_scales=(0.85, 1.25),

        # Hybrid refine
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
data_root = "/root/autodl-tmp/data/S3DIS"

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
            dict(type="RandomRotateTargetAngle", angle=(1 / 2, 1, 3 / 2), center=[0, 0, 0], axis="z", p=0.75),
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
                keys=("coord", "color", "normal", "segment"),
                return_min_coord=True,
            ),
            dict(type="SphereCrop", point_max=max_input_pts, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "segment"), feat_keys=("coord", "color", "normal")),
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
                keys=("coord", "color", "normal", "segment"),
                return_min_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "segment"), feat_keys=("coord", "color", "normal")),
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
                keys=("coord", "color", "normal"),
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "index"), feat_keys=("coord", "color", "normal")),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1)],
                [
                    dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)
