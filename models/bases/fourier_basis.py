"""Fixed public Fourier dictionary, independent of sample omega."""

import math
import torch
from torch import nn


class FourierBasis(nn.Module):
    def __init__(self, harmonics=16, frequency_step=math.pi):
        super().__init__()
        self.frequencies = tuple(range(1, harmonics + 1))
        self.frequency_step = frequency_step

    def derivatives(self, x):
        if x.ndim != 2 or x.shape[1] != 1:
            raise ValueError('x must have shape [N,1]')
        n = torch.tensor(self.frequencies, dtype=x.dtype, device=x.device)
        frequencies = n[None, :] * self.frequency_step
        phase = x * frequencies
        sine, cosine = torch.sin(phase), torch.cos(phase)
        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)
        phi = torch.stack((sine, cosine), dim=-1).flatten(1)
        d1 = torch.stack((frequencies * cosine, -frequencies * sine), dim=-1).flatten(1)
        d2 = torch.stack((-frequencies.square() * sine, -frequencies.square() * cosine), dim=-1).flatten(1)
        return (torch.cat((ones, phi), dim=1), torch.cat((zeros, d1), dim=1),
                torch.cat((zeros, d2), dim=1))

    def forward(self, x):
        return self.derivatives(x)[0]
