from __future__ import annotations

from abc import abstractmethod
from typing import Optional
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from ..dataops.dataset import DatasetConfig, DatasetPartition, DatasetOutput, ImageDataset, OfficeHomeDataset, PACSDataset, DigitsDGDataset, SupportedDatasets
from ..dataops.sampler import DomainBatchSampler

__all__ = ['BaseDataModule']


class BaseDataModule(pl.LightningDataModule):
    """
    Parameters
    ----------
    dataset_config : DatasetConfig
        The dataset parameters to run.
    batch_size : int
        The number of batches to
        train/validate the data before terminating.
        Defaults to 1.
    num_workers: int | None
        The number of workers to load the dataset.
        Defaults to 1 if not provided.
    """
    def __init__(self, dataset_config: DatasetConfig, batch_size: int = 1, num_workers: Optional[int] = 1) -> None:
        super().__init__()

        self.ds_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = 1 if num_workers is None else num_workers
        self.train_ds: ImageDataset
        self.test_ds: ImageDataset
        self.val_ds: ImageDataset
        self.full_ds: ImageDataset
        self.train_batch_sampler: DomainBatchSampler
        self.val_batch_sampler: DomainBatchSampler
        self.test_batch_sampler: DomainBatchSampler
        self.full_batch_sampler: DomainBatchSampler

    def setup(self, stage: str):
        ds_name = self.ds_config.dataset_name
        if ds_name == SupportedDatasets.PACS:
            dataset_cls = PACSDataset
        elif ds_name == SupportedDatasets.DIGITS:
            dataset_cls = DigitsDGDataset
        elif ds_name == SupportedDatasets.OFFICE:
            dataset_cls = OfficeHomeDataset
        else:
            raise ValueError(f'Unsupported dataset with name {ds_name}')

        if stage == 'fit':
            self.train_ds = dataset_cls(self.ds_config, partition=DatasetPartition.TRAIN)
            self.val_ds = dataset_cls(self.ds_config, partition=DatasetPartition.VALIDATE)
            self.train_batch_sampler = DomainBatchSampler(sampler=None, batch_size=self.batch_size, drop_last=False, image_dataset=self.train_ds)
            self.val_batch_sampler = DomainBatchSampler(sampler=None, batch_size=self.batch_size, drop_last=False, image_dataset=self.val_ds)
        elif stage == 'test':
            if ds_name is SupportedDatasets.DIGITS:
                self.test_ds = dataset_cls(self.ds_config, partition=DatasetPartition.VALIDATE)
            else:
                self.test_ds = dataset_cls(self.ds_config, partition=DatasetPartition.TEST)
            self.test_batch_sampler = DomainBatchSampler(sampler=None, batch_size=self.batch_size, drop_last=False, image_dataset=self.test_ds)
        else:
            self.full_ds = dataset_cls(self.ds_config, partition=DatasetPartition.ALL)
            self.full_batch_sampler = DomainBatchSampler(sampler=None, batch_size=self.batch_size, drop_last=False, image_dataset=self.full_ds)

    @abstractmethod
    def train_dataloader(self) -> DataLoader[DatasetOutput]:
        raise NotImplementedError

    @abstractmethod
    def val_dataloader(self) -> DataLoader[DatasetOutput]:
        raise NotImplementedError

    @abstractmethod
    def test_dataloader(self) -> DataLoader[DatasetOutput]:
        raise NotImplementedError

    @abstractmethod
    def predict_dataloader(self) -> DataLoader[DatasetOutput]:
        raise NotImplementedError
