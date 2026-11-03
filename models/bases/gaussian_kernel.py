"""Gaussian kernels over a learned positional encoding."""

import torch
from torch import nn

from models.bases.base import column_derivatives


class PositionalEncoder(nn.Module):
    def __init__(self, n_freqs=3, width=512):
        super().__init__()
        self.n_freqs = n_freqs
        self.network = nn.Sequential(nn.Linear(1 + 2*n_freqs, width), nn.ReLU(),
                                     nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))

    def forward(self, x):
        bands = torch.linspace(1.0, 2.0 ** (self.n_freqs - 1), self.n_freqs,
                               device=x.device, dtype=x.dtype)
        features = [x]
        for band in bands:
            features += [torch.sin(band*x), torch.cos(band*x)]
        return self.network(torch.cat(features, dim=1))


class GaussianKernelBasis(nn.Module):
    def __init__(self, anchors, sigma=0.05, n_freqs=3, learn_sigma=False,
                 identity_encoder=False):
        super().__init__()
        if anchors.ndim != 2 or anchors.shape[1] != 1 or torch.unique(anchors).numel() != anchors.shape[0]:
            raise ValueError('Gaussian anchors must be distinct [K,1] points')
        self.register_buffer('anchors', anchors.clone())
        self.sigma = nn.Parameter(torch.tensor(float(sigma), dtype=anchors.dtype), requires_grad=learn_sigma)
        self.encoder = (nn.Identity() if identity_encoder else PositionalEncoder(n_freqs).to(dtype=anchors.dtype))

    def forward(self, x):
        centers = self.encoder(self.anchors)
        queries = self.encoder(x)
        squared = (queries[:, None, :] - centers[None, :, :]).square().sum(dim=-1)
        return torch.exp(-squared / (2 * self.sigma.square()))

    def derivatives(self, x):
        return column_derivatives(self, x)
