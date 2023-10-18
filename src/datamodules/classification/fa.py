from __future__ import annotations

from typing import Protocol

from torch.utils.data import DataLoader

from ...tasks.classification import FA_X, Classification_Y

__all__ = ['FADataModule']


class FADataModule(Protocol):
    def train_dataloader(self) -> DataLoader[tuple[FA_X, Classification_Y]]: ...
    def val_dataloader(self) -> DataLoader[tuple[FA_X, Classification_Y]]: ...
    def test_dataloader(self) -> DataLoader[tuple[FA_X, Classification_Y]]: ...
    def predict_dataloader(self) -> DataLoader[FA_X]: ...


# DataLoader[tuple[torch.Tensor, ...]]
class ConcreteFADataModule(FADataModule):
    def __init__(self, *, max_batches: int | None = None, **kwargs) -> None: ...
