"""Frozen PriorEncoderHP definition (TaxoFunc-P2FM-ens-v2)."""
import torch.nn as nn


class PriorEncoderHP(nn.Module):
    def __init__(self, in_dim=128, hidden=256, out_dim=100, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.GELU(), nn.LayerNorm(hidden)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
