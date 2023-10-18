from __future__ import annotations

from typing import Protocol

from torch.utils.data import DataLoader

from ...tasks.nst import StyleTransfer_X

__all__ = ['AdaINDataModule']


class AdaINDataModule(Protocol):
    def train_dataloader(self) -> DataLoader[StyleTransfer_X]: ...
    def val_dataloader(self) -> DataLoader[StyleTransfer_X]: ...
    def test_dataloader(self) -> DataLoader[StyleTransfer_X]: ...
    def predict_dataloader(self) -> DataLoader[StyleTransfer_X]: ...


# DataLoader[tuple[torch.Tensor, ...]]
class ConcreteAdaINDataModule(AdaINDataModule):
    def __init__(self, *, max_batches: int | None = None, **kwargs) -> None: ...
