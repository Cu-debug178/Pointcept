_base_ = ["../_base_/default_runtime.py"]

# Settings for 4GB GPU (GTX 1650 Ti)
num_worker = 1
batch_size = 1
mix_prob = 0.0
max_input_pts = 16000

# model settings
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="kpconvx_base",
        input_channels=9,
        num_classes=13,
        dim=3,
        task='cloud_segmentation',
        kp_mode='kpconvx',
        shell_sizes=(1, 14, 28),
        kp_radius=2.3,
        kp_aggregation='nearest',
        kp_influence='constant',
        kp_sigma=2.3,
        share_kp=False,
        conv_groups=-1,
        inv_groups=8,
        inv_act='sigmoid',
        inv_grp_norm=True,
        kpx_upcut=False,
        subsample_size=0.02,
        neighbor_limits=(12, 16, 20, 20, 20),
        layer_blocks=(2, 2, 6, 8, 2),
        init_channels=32,
        channel_scaling=1.414,
        radius_scaling=2.2,
        decoder_layer=True,
        grid_pool=True,
        upsample_n=3,
        first_inv_layer=1,
        drop_path_rate=0.2,
        norm='batch',
        bn_momentum=0.1,
        smooth_labels=False,
        class_w=(),
    ),
    criteria=[
        dict(type="CrossEntropyLoss",
             loss_weight=1.0,
             ignore_index=-1),
    ]
)

# scheduler settings
epoch = 10
eval_epoch = 5
optimizer = dict(type="AdamW", lr=0.003, weight_decay=0.01)
scheduler = dict(type="OneCycleLR",
                 max_lr=optimizer["lr"],
                 pct_start=0.05,
                 anneal_strategy="cos",
                 div_factor=100.0,
                 final_div_factor=1000.0)

# dataset settings - S3DIS
dataset_type = "S3DISDataset"
data_root = "/root/autodl-tmp/data/S3DIS"

data = dict(
    num_classes=13,
    ignore_index=-1,
    names=["ceiling", "floor", "wall", "beam", "column", "window",
           "door", "table", "chair", "sofa", "bookcase", "board", "clutter"],
    train=dict(
        type=dataset_type,
        split=["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"],
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="GridSample",
                 grid_size=0.02,
                 hash_type="fnv",
                 mode="train",
                 keys=("coord", "color", "normal", "segment"),
                 return_min_coord=True),
            dict(type="SphereCrop", point_max=max_input_pts, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "segment"), feat_keys=("coord", "color", "normal"))
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="Area_5",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="GridSample",
                 grid_size=0.02,
                 hash_type="fnv",
                 mode="train",
                 keys=("coord", "color", "normal", "segment"),
                 return_min_coord=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "segment"), feat_keys=("coord", "color", "normal"))
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
            voxelize=dict(type="GridSample",
                          grid_size=0.02,
                          hash_type="fnv",
                          mode="test",
                          keys=("coord", "color", "normal")
                          ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "index"), feat_keys=("coord", "color", "normal"))
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)],
            ]
        )
    ),
)
