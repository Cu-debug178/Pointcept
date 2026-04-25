import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/root/autodl-tmp/Pointcept')

from pointcept.engines.defaults import default_config_parser
from pointcept.utils.checkpoint import load_checkpoint
from pointcept.models import build_model
from pointcept.datasets import build_dataset
from torch.utils.data import DataLoader

# Load config
cfg = default_config_parser('/root/autodl-tmp/Pointcept/configs/s3dis/semseg-kpconvx-hybrid-sgca-da-refine-v1m1-0-area5.py', [])

# Build dataset
dataset = build_dataset(cfg.data.train)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=cfg.data.train.get('batch_size', 3),
    shuffle=True,
    num_workers=cfg.data.train.get('num_worker', 8),
    collate_fn=dataset.collate_fn,
    pin_memory=True,
    drop_last=True,
)

# Build model
model = build_model(cfg.model)
model = model.cuda()
model.train()

# Load checkpoint
load_checkpoint(
    model,
    '/root/autodl-tmp/pointcept/exp/s3dis/s3dis-kpconvx-hybrid-training/model/model_last.pth',
    map_location='cuda'
)

# Run one batch for debugging
for batch in dataloader:
    print('=== Batch Info ===')
    print('Keys:', list(batch.keys()))
    
    # Move to GPU
    input_dict = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            input_dict[key] = value.cuda(non_blocking=True)
        else:
            input_dict[key] = value
    
    # Check segment values
    if 'segment' in input_dict:
        seg = input_dict['segment']
        print(f'Segment shape: {seg.shape}')
        print(f'Segment min: {seg.min().item()}, max: {seg.max().item()}')
        print(f'Segment unique values: {torch.unique(seg)}')
        print(f'Number of -1 values: {(seg == -1).sum().item()}')
    
    # Try forward pass with debugging
    try:
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            output_dict = model(input_dict)
            loss = output_dict['loss']
        print('Forward pass succeeded')
        print(f'Loss: {loss.item()}')
    except RuntimeError as e:
        print(f'Error during forward pass: {e}')
        import traceback
        traceback.print_exc()
    break
