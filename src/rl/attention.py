"""Temporal attention mechanism placeholder."""
import torch.nn as nn
class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads=4)
