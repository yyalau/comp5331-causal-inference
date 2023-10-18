from __future__ import annotations

from typing import Protocol

from torch.utils.data import DataLoader

from ...tasks.classification import ERM_X, Classification_Y

__all__ = ['ERMDataModule']


class ERMDataModule(Protocol):
    def train_dataloader(self) -> DataLoader[tuple[ERM_X, Classification_Y]]: ...
    def val_dataloader(self) -> DataLoader[tuple[ERM_X, Classification_Y]]: ...
    def test_dataloader(self) -> DataLoader[tuple[ERM_X, Classification_Y]]: ...
    def predict_dataloader(self) -> DataLoader[ERM_X]: ...


# DataLoader[tuple[torch.Tensor, ...]]
class ConcreteERMDataModule(ERMDataModule):
    def __init__(self, *, max_batches: int | None = None, **kwargs) -> None: ...
