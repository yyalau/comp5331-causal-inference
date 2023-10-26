from __future__ import annotations

import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np

__all__ = ['FourierMix']


class FourierMix(nn.Module):
    """
    Applies the "*amplitude mix*" strategy to produce a stylized image from a
    content image (`x`) and a style image (`x_prime`).

    Parameters
    ----------
    eta : float, default 2.0
        A hyperparameter controlling the maximum style mixing rate.
    """
    def __init__(self, eta: float = 0.5) -> None:
        super().__init__()

        self.eta = eta

    def forward(self, x: torch.Tensor, x_prime: torch.Tensor):
        # Input image shape: (batch_size, num_channels, depth, height, width)
        assert x.shape == x_prime.shape

        x_fft = fft.fft2(x, dim=(-2, -1))
        x_prime_fft = fft.fft2(x_prime, dim=(-2, -1))

        x_amp, x_pha = torch.abs(x_fft), torch.angle(x_fft)
        x_prime_amp, x_prime_pha = torch.abs(x_prime_fft), torch.angle(x_prime_fft)

        x_amp = fft.fftshift(x_amp, dim=(-2, -1))
        x_prime_amp = fft.fftshift(x_prime_amp, dim=(-2, -1))

        x_amp = torch.clone(x_amp)
        x_prime_amp = torch.clone(x_prime_amp)

        lambda_ = np.random.uniform(0, self.eta)

        x_hat = (1 - lambda_) * x_amp + lambda_ * x_prime_amp
        x_hat = fft.ifftshift(x_hat, dim=(-2, -1))
        x_hat = x_hat * (torch.exp(1j * x_pha))
        x_hat = torch.real(fft.ifft2(x_hat, dim=(-2, -1)))

        return x_hat
