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
import torchvision.transforms.v2 as T

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
    PACS = 'pacs'
    OFFICE = 'OfficeHomeDataset_10072016'
    DIGITS = 'digits_dg'

class DatasetOutput(BaseModel):
    image: Tensor
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
    rand_augment: List[int]
    resize_height: int
    resize_width: int
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
            raise ValueError('not supported dataset {cls.dataset_name}')

    def __init__(self, config: DatasetConfig, partition: DatasetPartition) -> None:
        super().__init__()
        self.validate_dataset_name()
        self.validate_domains(config.train_val_domains)
        self.validate_domains(config.test_domains)
        self.lazy = config.lazy
        self.partition = partition
        self.num_domains_to_sample = config.num_domains_to_sample
        self.num_ood_samples = config.num_ood_samples
        self.dataset_path_root = config.dataset_path_root

        if not config.dataset_path_root.exists():
            parent_root = config.dataset_path_root.parent.name
            self.download(parent_root)
            config.dataset_path_root = Path(f'{parent_root}/{self.dataset_name}')

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
        assert len(self.rand_augment) == 2
        self.transforms = RandAugment(*self.rand_augment)
        self.height = config.resize_height
        self.width = config.resize_width

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

    def _preprocess(self, X: npt.NDArray[np.float32]) -> Tensor:

        image = Image.fromarray(X)

        transform = T.Compose([
            T.Resize((self.height, self.width), interpolation=T.InterpolationMode.BILINEAR),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ])

        resized_img = transform(self.transforms(image))
        assert isinstance(resized_img, Tensor)

        return resized_img

    def _ood_sample(
        self, domain_list: List[str], num_domains_to_sample: int, num_ood_samples: int
    ) -> Tensor:
        domain_data_map = copy.deepcopy(self.domain_data_map)

        style = [
            sample.load()
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

        return torch.stack([self._preprocess(s) for s in style])

    def _create_tensors_from_batch(
        self, batch: List[DatasetOutput]
    ) -> Tuple[Tensor, Tensor, List[str]]:
        content = torch.stack([data.image for data in batch])
        labels = torch.from_numpy((np.array([data.label for data in batch])))
        domains = [data.domain for data in batch]
        return content, labels, domains

    def num_domains(self) -> int:
        return len(self.domain_data_map.keys())

    def collate_erm(self, batch: List[DatasetOutput]) -> Tuple[ERM_X, Classification_Y]:
        content, labels, _ = self._create_tensors_from_batch(batch)
        return content, labels

    def collate_st(self, batch: List[DatasetOutput]) -> StyleTransfer_X:
        content, _, domains = self._create_tensors_from_batch(batch)
        style = self._ood_sample(domains, 1, 1)
        assert len(style) == len(batch)
        return StyleTransfer_X(content=content, style=style)

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
