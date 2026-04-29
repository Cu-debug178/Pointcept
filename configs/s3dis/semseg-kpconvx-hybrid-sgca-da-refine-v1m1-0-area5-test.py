_base_ = ["./semseg-kpconvx-hybrid-sgca-da-refine-v1m1-0-area5.py"]

# Standalone test config for a trained KPConvX-Hybrid checkpoint.
# This keeps the official full-scene testing logic, but splits each large
# test fragment into smaller sub-fragments and merges logits back by index.

empty_cache = True
batch_size_test = 1
fragment_batch_size_test = 1 #总占用4G显存，和max_input_pts=80000匹配
# fragment_batch_size_test = 2

# test-only process
evaluate = False
test_only = True

data = dict(
    
    test=dict(
        test_cfg=dict(
            # crop=dict(_delete_=True, type="TestSphereCrop", 
            # point_max=160000),
            crop=dict(_delete_=True, type="TestSphereCrop", 
            point_max=80000),   #占用4G显存，和fragment_batch_size_test=1匹配
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", 
                angle=[0], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", 
                angle=[1 / 2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", 
                angle=[1], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", 
                angle=[3 / 2], axis="z", center=[0, 0, 0], p=1)],
            ],
        )
    )
)
