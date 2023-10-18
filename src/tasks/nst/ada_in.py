from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ...models.nst import StyleTransfer_X, StyleTransfer_Y, AdaINDecoder, AdaINEncoder

from ..base import BaseTask

__all__ = ['AdaINTask', 'StyleTransfer_X', 'StyleTransfer_Y']


class AdaINTask(BaseTask[StyleTransfer_X, StyleTransfer_X, StyleTransfer_Y]):
    def __init__(
        self,
        encoder: AdaINEncoder,
        decoder: AdaINDecoder,
        *,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], Optimizer],
        scheduler: Callable[[Optimizer], LRScheduler],
        alpha: float = 1.0,
        gamma: float = 2.0,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
        )

        self.encoder = encoder
        self.decoder = decoder

        self.alpha = alpha
        self.gamma = gamma

        self.loss = self._combined_loss
        self.metrics = {
            'content_loss': self._content_loss,
            'style_loss': self._style_loss,
        }

    def _content_loss_fn(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(input_, target)

    def _style_loss_fn(self, input_states: list[torch.Tensor], target_states: list[torch.Tensor]) -> torch.Tensor:
        std_loss, mean_loss = torch.tensor(0.0), torch.tensor(0.0)

        for input_state, target_state in zip(input_states, target_states):
            input_std, input_mean = torch.std_mean(input_state, dim=(-2, -1))
            target_std, target_mean = torch.std_mean(target_state, dim=(-2, -1))

            std_loss += F.mse_loss(input_std, target_std)
            mean_loss += F.mse_loss(input_mean, target_mean)

        return mean_loss + std_loss

    def _content_loss(self, enc_applied_states: list[torch.Tensor], enc_style_states: list[torch.Tensor], content: torch.Tensor) -> torch.Tensor:
        return self._content_loss_fn(enc_applied_states[-1], content)

    def _style_loss(self, enc_applied_states: list[torch.Tensor], enc_style_states: list[torch.Tensor], content: torch.Tensor) -> torch.Tensor:
        return self._style_loss_fn(enc_applied_states, enc_style_states)

    def _combined_loss(self, enc_applied_states: list[torch.Tensor], enc_style_states: list[torch.Tensor], content: torch.Tensor) -> torch.Tensor:
        content_loss = self._content_loss(enc_applied_states, enc_style_states, content)
        style_loss = self._style_loss(enc_applied_states, enc_style_states, content)
        gamma = self.gamma

        return content_loss + gamma * style_loss

    def _ada_in(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        content_std, content_mean = torch.std_mean(content, dim=(-2, -1))
        style_std, style_mean = torch.std_mean(style, dim=(-2, -1))

        return style_std * (content - content_mean) / content_std + style_mean

    def _eval_step(self, batch: StyleTransfer_X, batch_idx: int) -> dict[str, torch.Tensor]:
        x = batch

        enc_style_states = self.encoder.get_states(x['style'])
        enc_content = self.encoder(x['content'])

        enc_applied = self._ada_in(enc_content, enc_style_states[-1])
        applied = self.decoder(enc_applied)

        enc_applied_states = self.encoder.get_states(applied)

        loss = self.loss(enc_applied_states, enc_style_states, enc_content)
        metrics = {name: metric(enc_applied_states, enc_style_states, enc_content) for name, metric in self.metrics.items()}

        return {'loss': loss, **metrics}

    def forward(self, batch: StyleTransfer_X) -> StyleTransfer_Y:
        enc_style = self.encoder(batch['style'])
        enc_content = self.encoder(batch['content'])
        alpha = self.alpha

        return alpha * self._ada_in(enc_content, enc_style) + (1 - alpha) * enc_content
