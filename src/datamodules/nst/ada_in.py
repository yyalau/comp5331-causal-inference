from __future__ import annotations

from torch.utils.data import DataLoader
from typing import Optional
from ...dataops.dataset import DatasetConfig, DatasetOutput

from ..base import BaseDataModule

__all__ = ['AdaINDataModule']


class AdaINDataModule(BaseDataModule):
    def __init__(self, dataset_config: DatasetConfig, batch_size: int = 1, num_workers: Optional[int] = 1) -> None:
        super().__init__(dataset_config, batch_size, num_workers)

    def train_dataloader(self) -> DataLoader[DatasetOutput]:
        return DataLoader(self.train_ds, num_workers=self.num_workers, collate_fn=self.train_ds.collate_st, batch_sampler=self.train_batch_sampler)

    def val_dataloader(self) -> DataLoader[DatasetOutput]:
        return DataLoader(self.val_ds, num_workers=self.num_workers, collate_fn=self.val_ds.collate_st, batch_sampler=self.val_batch_sampler)

    def test_dataloader(self) -> DataLoader[DatasetOutput]:
        return DataLoader(self.test_ds, num_workers=self.num_workers, collate_fn=self.test_ds.collate_st, batch_sampler=self.test_batch_sampler)

    def predict_dataloader(self) -> DataLoader[DatasetOutput]:
        return DataLoader(self.full_ds, num_workers=self.num_workers, collate_fn=self.full_ds.collate_st, batch_sampler=self.full_batch_sampler)
