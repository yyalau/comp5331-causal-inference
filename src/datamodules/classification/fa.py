from __future__ import annotations

from torch.utils.data import DataLoader
import lightning.pytorch as pl

from ...tasks.classification import FA_X, Classification_Y

__all__ = ['FADataModule']


class FADataModule(pl.LightningDataModule):
    def __init__(self, *, max_batches: int | None = None, **kwargs) -> None:
        super().__init__()

        raise NotImplementedError

    def train_dataloader(self) -> DataLoader[tuple[FA_X, Classification_Y]]:
        raise NotImplementedError

    def val_dataloader(self) -> DataLoader[tuple[FA_X, Classification_Y]]:
        raise NotImplementedError

    def test_dataloader(self) -> DataLoader[tuple[FA_X, Classification_Y]]:
        raise NotImplementedError

    def predict_dataloader(self) -> DataLoader[FA_X]:
        raise NotImplementedError
