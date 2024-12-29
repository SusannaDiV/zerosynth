# test_tensorboard.py
from torch.utils.tensorboard import SummaryWriter
import torch

# Create a test writer
writer = SummaryWriter('runs/original_training/test_run')

# Log some test data
for i in range(100):
    writer.add_scalar('test_loss', torch.randn(1).item(), i)

writer.close()