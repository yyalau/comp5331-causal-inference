from __future__ import annotations

import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np

__all__ = ['fourierMix']


class fourierMix(nn.Module):
    def __init__(self, eta: float = 0.5) -> None:
        super().__init__()
        self.eta = eta
    
    def forward(self, x, x_prime):
        # Input image shape: (batch_size, num_channels, depth, height, width)

        assert x.shape == x_prime.shape

        x_fft = fft.fft2(x)
        x_prime_fft = fft.fft2(x_prime)

        x_amp, x_pha = torch.abs(x_fft), torch.angle(x_fft)
        x_prime_amp, x_prime_pha = torch.abs(x_prime_fft), torch.angle(x_prime_fft)

        x_amp = fft.fftshift(x_amp)
        x_prime_amp = fft.fftshift(x_prime_amp)

        x_amp = torch.clone(x_amp)
        x_prime_amp = torch.clone(x_prime_amp)

        lamda = np.random.uniform(0, self.eta)

        x_hat = (1-lamda)*x_amp + lamda*x_prime_amp
        x_hat = fft.ifftshift(x_hat)
        x_hat = x_hat * (torch.exp(1j * x_pha))
        x_hat = torch.real(fft.ifft2(x_hat))

        return x_hat
