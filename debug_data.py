import os
import sys
sys.path.insert(0, '/root/autodl-tmp/Pointcept')

import numpy as np
import torch

from pointcept.datasets import build_dataset
from pointcept.datasets.transform import Compose

# Load dataset
cfg = dict(
    type="S3DISDataset",
    split=["Area_1", "Area_2", "Area_3", "Area_4", "Area_6"],
    data_root="/root/autodl-tmp/data/s3dis",
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
            return_min_coord=True,
        ),
        dict(type="SphereCrop", point_max=30000, mode="random"),
        dict(type="CenterShift", apply_z=False),
        dict(type="NormalizeColor"),
        dict(type="ShufflePoint"),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "segment"), feat_keys=("coord", "color", "normal")),
    ],
    test_mode=False,
)

dataset = build_dataset(cfg)

# Check first few samples
for i in range(5):
    data = dataset[i]
    print(f"\n=== Sample {i} ===")
    print(f"Keys: {list(data.keys())}")
    if "segment" in data:
        seg = data["segment"]
        print(f"Segment shape: {seg.shape}")
        print(f"Segment dtype: {seg.dtype}")
        print(f"Segment min: {seg.min().item()}, max: {seg.max().item()}")
        print(f"Segment unique: {torch.unique(seg)}")
        print(f"Number of -1 values: {(seg == -1).sum().item()}")
    if "coord" in data:
        print(f"Coord shape: {data['coord'].shape}")
    if "offset" in data:
        print(f"Offset: {data['offset']}")

# Check collate function
from pointcept.datasets import collate_fn, point_collate_fn

# Get a batch
batch_data = [dataset[i] for i in range(3)]
collated = point_collate_fn(batch_data)
print("\n=== Collated Batch ===")
print(f"Keys: {list(collated.keys())}")
if "segment" in collated:
    seg = collated["segment"]
    print(f"Segment shape: {seg.shape}")
    print(f"Segment min: {seg.min().item()}, max: {seg.max().item()}")
    print(f"Unique values: {torch.unique(seg)}")
    print(f"Number of -1 values: {(seg == -1).sum().item()}")
if "offset" in collated:
    print(f"Offset: {collated['offset']}")
    # Check if offset is correct
    offsets = collated["offset"]
    prev = 0
    for j, off in enumerate(offsets):
        print(f"  Batch {j}: {prev} - {off.item()} ({off.item() - prev} points)")
        prev = off.item()
