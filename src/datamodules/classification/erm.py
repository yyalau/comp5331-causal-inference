from __future__ import annotations

from torch.utils.data import DataLoader
import lightning.pytorch as pl

from ...tasks.classification import ERM_X, Classification_Y

__all__ = ['ERMDataModule']


class ERMDataModule(pl.LightningDataModule):
    def __init__(self, *, max_batches: int | None = None, **kwargs) -> None:
        super().__init__()

        raise NotImplementedError

    def train_dataloader(self) -> DataLoader[tuple[ERM_X, Classification_Y]]:
        raise NotImplementedError

    def val_dataloader(self) -> DataLoader[tuple[ERM_X, Classification_Y]]:
        raise NotImplementedError

    def test_dataloader(self) -> DataLoader[tuple[ERM_X, Classification_Y]]:
        raise NotImplementedError

    def predict_dataloader(self) -> DataLoader[ERM_X]:
        raise NotImplementedError
