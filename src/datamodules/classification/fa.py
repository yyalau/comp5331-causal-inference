from __future__ import annotations

from torch.utils.data import DataLoader
from typing import Optional
from ...dataops.dataset import DatasetConfig, DatasetOutput

from ..base import BaseDataModule

__all__ = ['FADataModule']


class FADataModule(BaseDataModule):
    def __init__(self, dataset_config: DatasetConfig, batch_size: Optional[int] = None, num_workers: Optional[int] = 1) -> None:
        super().__init__(dataset_config, batch_size, num_workers)

    def train_dataloader(self) -> DataLoader[DatasetOutput]:
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.train_ds.collate_fa)

    def val_dataloader(self) -> DataLoader[DatasetOutput]:
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.val_ds.collate_fa)

    def test_dataloader(self) -> DataLoader[DatasetOutput]:
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.test_ds.collate_fa)

    def predict_dataloader(self) -> DataLoader[DatasetOutput]:
        return DataLoader(self.full_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, collate_fn=self.full_ds.collate_fa)
