from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
import copy
from dataclasses import dataclass
from enum import Enum
from ..func import (
    get_flattened_index,
    sample_dictionary,
    sample_sequence_delete,
    sample_sequence_no_replace
)

import numpy as np

from pathlib import Path

# from PIL import Image
from pydantic import BaseModel

from torch.utils.data import Dataset
import torch

from typing import List, Optional, Tuple
from typing_extensions import TypeAlias

from ...tasks.classification import Classification_Y, FA_X, ERM_X
from ...tasks.nst import StyleTransfer_X

from ..utils import download_from_gdrive, unzip

from ..image import ImageLoader, PreprocessParams
from functools import reduce
import operator
__all__ = [
    "ImageDataset",
    "SupportedDatasets",
    "DatasetPartition",
    "DatasetConfig",
    "DatasetOutput",
]

Tensor: TypeAlias = torch.Tensor


class SupportedDatasets(str, Enum):
    PACS = "pacs"
    OFFICE = "OfficeHomeDataset_10072016"
    DIGITS = "digits_dg"


class DatasetOutput(BaseModel):
    image: Tensor
    label: int
    domain: str

    class Config:
        arbitrary_types_allowed = True


@dataclass(frozen=False)
class DatasetConfig:
    """
    Parameters
    ----------
    dataset_path_root : Path
        The path to the root directory containing the data.
    dataset_name : SupportedDatasets
        The name of the dataset.
    train_val_domains : List[str]
        The domains to use for training and validation.
    test_domains: List[str]
        The domains to use for testing
    lazy : bool
        Lazy initialization of the images.
    preprocess_params : PreprocessParams
        Parameters for preprocessing the image.
    k : Optional[int]:
        Size of samples for style images
    """

    dataset_path_root: Path
    dataset_name: SupportedDatasets
    train_val_domains: List[str]
    test_domains: List[str]
    lazy: bool
    preprocess_params: PreprocessParams
    k: Optional[int]

class DatasetPartition(str, Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATE = "val"
    ALL = "full"


@dataclass
class ImageReader:
    load: ImageLoader
    label: int

class ImageDataset(Dataset[DatasetOutput]):
    data_url = ""
    dataset_name = ""

    @abstractmethod
    def _fetch_data(self) -> Mapping[str, List[ImageReader]]:
        pass

    @classmethod
    @abstractmethod
    def validate_domains(cls, domains: List[str]) -> None:
        raise NotImplementedError()

    @classmethod
    def download(cls, destination: str) -> None:
        print(f"Downloading data from {cls.data_url}")

        file_path = download_from_gdrive(
            cls.data_url, f"{destination}/{cls.dataset_name}.zip"
        )
        print(f"Extracting files from {file_path}")
        unzip(file_path)

    @classmethod
    def validate_dataset_name(cls):
        try:
            SupportedDatasets(cls.dataset_name)
        except ValueError:
            raise ValueError("not supported dataset {cls.dataset_name}")

    def _validate_config_params(self, config: DatasetConfig):
        self.validate_dataset_name()
        self.validate_domains(config.train_val_domains)
        self.validate_domains(config.test_domains)

    def __init__(self, config: DatasetConfig, partition: DatasetPartition) -> None:
        super().__init__()
        self._validate_config_params(config)

        self.lazy = config.lazy
        self.partition = partition
        self.k = config.k
        self.dataset_path_root = config.dataset_path_root
        self.preprocessor_params = config.preprocess_params
        if not config.dataset_path_root.exists():
            parent_root = config.dataset_path_root.parent.name
            self.download(parent_root)
            config.dataset_path_root = Path(f"{parent_root}/{self.dataset_name}")

        self.domains = (
            config.test_domains
            if partition is DatasetPartition.TEST
            else config.train_val_domains
        )
        self.domain_data_map = self._fetch_data()

        self.len = sum(
            len(image_loader) for image_loader in self.domain_data_map.values()
        )


    def __getitem__(self, idx: int) -> DatasetOutput:
        """
        Get the preprocessed item at the specified index.
        """
        domain, item = get_flattened_index(self.domain_data_map, idx)
        processed_image = item.load()
        label = item.label
        return DatasetOutput(image=processed_image, label=label, domain=domain)

    def __len__(self) -> int:
        return self.len

    def _ood_sample(self, batch_size: int) -> Tensor:
        pool = list(reduce(operator.concat, self.domain_data_map.values()))
        samples = sample_sequence_no_replace(pool, batch_size)
        return torch.stack([
            image_reader.load()
            for image_reader in samples
        ])


    def _create_tensors_from_batch(
        self, batch: List[DatasetOutput]
    ) -> Tuple[Tensor, Tensor]:
        content = torch.stack([data.image for data in batch])
        labels = torch.from_numpy((np.array([data.label for data in batch])))
        return content, labels

    def num_domains(self) -> int:
        return len(self.domain_data_map.keys())

    def collate_erm(self, batch: List[DatasetOutput]) -> Tuple[ERM_X, Classification_Y]:
        content, labels = self._create_tensors_from_batch(batch)
        return content, labels

    def collate_st(self, batch: List[DatasetOutput]) -> StyleTransfer_X:
        batch_size = len(batch)
        content, _ = self._create_tensors_from_batch(batch)
        style = self._ood_sample(batch_size)
        return StyleTransfer_X(content=content, style=style)

    def collate_fa(
        self,
        batch: List[DatasetOutput],
    ) -> Tuple[FA_X, Classification_Y]:
        batch_size = len(batch)
        k = self.k
        if k is None:
            raise ValueError("Sample size `k` has not been set")
        content, labels = self._create_tensors_from_batch(batch)
        styles = [self._ood_sample(batch_size) for _ in range(k)]
        return FA_X(content=content, styles=styles), labels
