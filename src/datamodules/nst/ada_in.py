from __future__ import annotations

from torch.utils.data import DataLoader
import lightning.pytorch as pl

from ...tasks.nst import StyleTransfer_X

__all__ = ['AdaINDataModule']


class AdaINDataModule(pl.LightningDataModule):
    def __init__(self, *, max_batches: int | None = None, **kwargs) -> None:
        super().__init__()

        raise NotImplementedError

    def train_dataloader(self) -> DataLoader[StyleTransfer_X]:
        raise NotImplementedError

    def val_dataloader(self) -> DataLoader[StyleTransfer_X]:
        raise NotImplementedError

    def test_dataloader(self) -> DataLoader[StyleTransfer_X]:
        raise NotImplementedError

    def predict_dataloader(self) -> DataLoader[StyleTransfer_X]:
        raise NotImplementedError
