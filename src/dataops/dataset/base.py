from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

from pathlib import Path

from PIL import Image
from pydantic import BaseModel

from torch.utils.data import Dataset
import torch

from typing import List, Optional, Tuple
from typing_extensions import TypeAlias

from ...tasks.classification import Classification_Y, FA_X, ERM_X
from ...tasks.nst import StyleTransfer_X

from ..augmentation import RandAugment
from ..func import (
    get_flattened_index,
    sample_dictionary,
    sample_sequence_and_remove_from_population,
)
from ..utils import download_from_gdrive, unzip


__all__ = ["ImageDataset", 'SupportedDatasets', "DatasetPartition", "DatasetConfig", "DatasetOutput"]

Tensor: TypeAlias = torch.Tensor

class SupportedDatasets(str, Enum):
    PACS = 'PACS'
    OFFICE = 'OfficeHome'
    Digits =  'DigitsDG'

class DatasetOutput(BaseModel):
    image: npt.NDArray[np.float32]
    label: int
    domain: str

    class Config:
        arbitrary_types_allowed = True


@dataclass(frozen=False)
class DatasetConfig:
    dataset_path_root: Path
    dataset_name: SupportedDatasets
    train_val_domains: List[str]
    test_domains: List[str]
    lazy: bool
    rand_augment: Tuple[float, float]
    num_domains_to_sample: Optional[int]
    num_ood_samples: Optional[int]


class DatasetPartition(str, Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATE = "val"
    ALL = 'full'


@dataclass
class ImageReader:
    load: Callable[[], npt.NDArray[np.float32]]
    label: int


class ImageDataset(Dataset[DatasetOutput]):
    data_url = ""
    dataset_name = ""

    def __init__(self, config: DatasetConfig, partition: DatasetPartition) -> None:
        super().__init__()

        self.lazy = config.lazy
        self.partition = partition
        self.num_domains_to_sample = config.num_domains_to_sample
        self.num_ood_samples = config.num_ood_samples
        self.dataset_path_root = config.dataset_path_root
        if not config.dataset_path_root.exists():
            data_path = self.download(config.dataset_path_root.parent.name)
            config.dataset_path_root = Path(data_path)

        self.domains = (
            config.test_domains
            if partition is DatasetPartition.TEST
            else config.train_val_domains
        )
        self.domain_data_map = self._fetch_data()

        self.len = sum(
            len(image_loader) for image_loader in self.domain_data_map.values()
        )
        self.rand_augment = config.rand_augment
        self.transforms = RandAugment(*self.rand_augment)

    def __getitem__(self, idx: int) -> DatasetOutput:
        """
        Get the preprocessed item at the specified index.
        """
        domain, item = get_flattened_index(self.domain_data_map, idx)
        processed_image = self._preprocess(item.load())
        label = item.label
        return DatasetOutput(image=processed_image, label=label, domain=domain)

    def __len__(self) -> int:
        return self.len

    def _preprocess(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        image = Image.fromarray(X)
        return np.array(self.transforms(image))

    def _ood_sample(
        self, domain_list: List[str], num_domains_to_sample: int, num_ood_samples: int
    ) -> List[Tensor]:
        domain_data_map = copy.deepcopy(self.domain_data_map)

        return [
            torch.from_numpy(sample.load())
            for domain in domain_list
            for ood_domain in sample_dictionary(
                domain_data_map,
                num_domains_to_sample,
                lambda other_domain: other_domain != domain,
            )
            for sample in sample_sequence_and_remove_from_population(
                domain_data_map[ood_domain], num_ood_samples
            )
        ]

    def _create_tensors_from_batch(
        self, batch: List[DatasetOutput]
    ) -> Tuple[Tensor, Tensor, List[str]]:
        content = torch.from_numpy(np.array([data.image for data in batch]))
        labels = torch.from_numpy((np.array([data.label for data in batch])))
        domains = [data.domain for data in batch]
        return content, labels, domains

    @abstractmethod
    def _fetch_data(self) -> Mapping[str, List[ImageReader]]:
        pass

    @classmethod
    def download(cls, destination: str) -> str:
        print(f"Downloading data from {cls.data_url}")

        file_path = download_from_gdrive(
            cls.data_url, f"{destination}/{cls.dataset_name}.zip"
        )
        print(f"Extracting files from {file_path}")

        return unzip(file_path)

    def num_domains(self) -> int:
        return len(self.domain_data_map.keys())

    def collate_erm(self, batch: List[DatasetOutput]) -> Tuple[ERM_X, Classification_Y]:
        content, labels, _ = self._create_tensors_from_batch(batch)
        return content, labels

    def collate_st(self, batch: List[DatasetOutput]) -> StyleTransfer_X:
        content, _, domains = self._create_tensors_from_batch(batch)
        style = self._ood_sample(domains, 1, 1)
        assert len(style) == len(batch)
        return StyleTransfer_X(content=content, style=style[0])

    def collate_fa(
        self,
        batch: List[DatasetOutput],
    ) -> Tuple[FA_X, Classification_Y]:
        num_domains_to_sample = self.num_domains_to_sample
        num_ood_samples = self.num_ood_samples

        if num_domains_to_sample is None or num_ood_samples is None:
            raise ValueError('Values for collate are empty')

        content, labels, domains = self._create_tensors_from_batch(batch)
        styles = self._ood_sample(
            domains,
            num_domains_to_sample,
            num_ood_samples
        )
        return FA_X(content=content, styles=styles), labels
