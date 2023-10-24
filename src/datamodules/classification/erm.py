from __future__ import annotations

from torch.utils.data import DataLoader

from ...dataops.dataset import DatasetConfig, DatasetOutput

from ..base import BaseDataModule

__all__ = ['ERMDataModule']


class ERMDataModule(BaseDataModule):
    def __init__(self, dataset_config: DatasetConfig, max_batches: int | None = None, **kwargs) -> None:
        super().__init__(dataset_config, max_batches)

    def train_dataloader(self) -> DataLoader[DatasetOutput]:
        return DataLoader(self.train_ds, batch_size=self.max_batches, shuffle=True, collate_fn=self.train_ds.collate_erm)

    def val_dataloader(self) -> DataLoader[DatasetOutput]:
        return DataLoader(self.val_ds, batch_size=self.max_batches, shuffle=True, collate_fn=self.train_ds.collate_erm)

    def test_dataloader(self) -> DataLoader[DatasetOutput]:
        return DataLoader(self.test_ds, batch_size=self.max_batches, collate_fn=self.train_ds.collate_erm)


    def predict_dataloader(self) -> DataLoader[DatasetOutput]:
        return DataLoader(self.full_ds, batch_size=self.max_batches, collate_fn=self.train_ds.collate_erm)
