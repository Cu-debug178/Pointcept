import os
import sys
sys.path.insert(0, '/root/autodl-tmp/Pointcept')

import torch
import importlib.util

# Load config
config_path = "/root/autodl-tmp/Pointcept/configs/s3dis/semseg-kpconvx-hybrid-sgca-da-refine-v1m1-0-area5.py"

# Read and parse Python config
spec = importlib.util.spec_from_file_location("config", config_path)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
config = {k: v for k, v in vars(config_module).items() if not k.startswith('_')}

# Build model
from pointcept.models import build_model
model = build_model(config['model'])
model.cuda()
model.eval()

# Create dummy input
batch_size = 3
num_points = 30000
total_points = batch_size * num_points

dummy_coord = torch.randn(total_points, 3).cuda()
dummy_feat = torch.randn(total_points, 9).cuda()  # coord + color + normal
dummy_segment = torch.randint(0, 13, (total_points,)).cuda()  # 13 classes
dummy_offset = torch.tensor([num_points * i for i in range(1, batch_size + 1)]).cuda()

input_dict = {
    'coord': dummy_coord,
    'feat': dummy_feat,
    'segment': dummy_segment,
    'offset': dummy_offset,
}

print("Input shapes:")
print(f"  coord: {dummy_coord.shape}")
print(f"  feat: {dummy_feat.shape}")
print(f"  segment: {dummy_segment.shape}")
print(f"  offset: {dummy_offset.shape}")
print(f"  segment min: {dummy_segment.min().item()}, max: {dummy_segment.max().item()}")

# Test forward pass
print("\nRunning forward pass...")
try:
    with torch.no_grad():
        output = model(input_dict)
    print("Forward pass successful!")
    print(f"Output keys: {list(output.keys())}")
    if 'loss' in output:
        print(f"Loss: {output['loss'].item()}")
    if 'seg_logits' in output:
        print(f"Seg_logits shape: {output['seg_logits'].shape}")
except Exception as e:
    print(f"Error during forward pass: {e}")
    import traceback
    traceback.print_exc()

# Test with training mode
print("\nTesting in training mode...")
model.train()
try:
    output = model(input_dict)
    print("Training forward pass successful!")
    print(f"Output keys: {list(output.keys())}")
    if 'loss' in output:
        print(f"Loss: {output['loss'].item()}")
except Exception as e:
    print(f"Error during training forward pass: {e}")
    import traceback
    traceback.print_exc()
