from __future__ import annotations

from abc import abstractmethod

from torch.utils.data import DataLoader
import lightning.pytorch as pl

from ..dataops.dataset import DatasetConfig, DatasetPartition, DatasetOutput, ImageDataset, OfficeHomeDataset, PACSDataset, DigitsDGDataset, SupportedDatasets

__all__ = ['BaseDataModule']


class BaseDataModule(pl.LightningDataModule):
    """
    Parameters
    ----------
    dataset_config : DatasetConfig
        The dataset parameters to run.
    max_batches : int | None
        The maximum number of batches to
        train/validate the data before terminating.
        Defaults to None.
    """
    def __init__(self, dataset_config: DatasetConfig, max_batches: int | None = None) -> None:
        super().__init__()

        self.ds_config = dataset_config
        self.batch_size = batch_size
        self.train_ds: ImageDataset
        self.test_ds: ImageDataset
        self.val_ds: ImageDataset
        self.full_ds: ImageDataset

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
        elif stage == 'test':
            if ds_name is SupportedDatasets.DIGITS:
                self.test_ds = dataset_cls(self.ds_config, partition=DatasetPartition.VALIDATE)
            else:
                self.test_ds = dataset_cls(self.ds_config, partition=DatasetPartition.TEST)
        else:
            self.full_ds = dataset_cls(self.ds_config, partition=DatasetPartition.ALL)

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
