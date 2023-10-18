from __future__ import annotations

from typing import Protocol

from torch.utils.data import DataLoader

from ....tasks.classification import FAST_X, Classification_Y

__all__ = ['FASTDataModule']


class FASTDataModule(Protocol):
    def train_dataloader(self) -> DataLoader[tuple[FAST_X, Classification_Y]]: ...
    def val_dataloader(self) -> DataLoader[tuple[FAST_X, Classification_Y]]: ...
    def test_dataloader(self) -> DataLoader[tuple[FAST_X, Classification_Y]]: ...
    def predict_dataloader(self) -> DataLoader[FAST_X]: ...


# DataLoader[tuple[torch.Tensor, ...]]
class ConcreteFASTDataModule(FASTDataModule):
    def __init__(self, *, max_batches: int | None = None, **kwargs) -> None: ...
