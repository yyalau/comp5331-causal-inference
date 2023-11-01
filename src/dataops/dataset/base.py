from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
import copy
from dataclasses import dataclass
from enum import Enum

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

from ..func import (
    get_flattened_index,
    sample_dictionary,
    sample_sequence_and_remove_from_population,
)
from ..utils import download_from_gdrive, unzip

from ..image import ImageLoader, PreprocessParams

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
    num_domains_to_sample : int
        The number of domains to sample from for
        each training, validation or testing sample.
        Note that the value of this param must be
        less than the number of domains listed in
        train_val_domains and test_domains.
    num_ood_samples : int
        Number of images to sample per ood domain.
    """

    dataset_path_root: Path
    dataset_name: SupportedDatasets
    train_val_domains: List[str]
    test_domains: List[str]
    lazy: bool
    preprocess_params: PreprocessParams
    num_domains_to_sample: Optional[int]
    num_ood_samples: Optional[int]


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
        self.num_domains_to_sample = config.num_domains_to_sample
        self.num_ood_samples = config.num_ood_samples
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

    def _ood_sample(
        self, domain_list: List[str], num_domains_to_sample: int, num_ood_samples: int
    ) -> List[Tensor]:
        domain_data_map = copy.deepcopy(self.domain_data_map)
        styles: List[Tensor] = []
        for domain in domain_list:
            domain_styles: List[Tensor] = []
            for ood_domain in sample_dictionary(
                domain_data_map,
                num_domains_to_sample,
                lambda other_domain: other_domain != domain,
            ):
                for sample in sample_sequence_and_remove_from_population(
                    domain_data_map[ood_domain], num_ood_samples
                ):
                    domain_styles.append((sample.load()))
            styles.append(torch.stack(domain_styles))
        return styles

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
        style = torch.concat(self._ood_sample(domains, 1, 1))
        return StyleTransfer_X(content=content, style=style)

    def collate_fa(
        self,
        batch: List[DatasetOutput],
    ) -> Tuple[FA_X, Classification_Y]:
        num_domains_to_sample, num_ood_samples = self._validate_ood_sample_size()
        content, labels, domains = self._create_tensors_from_batch(batch)
        styles = self._ood_sample(domains, num_domains_to_sample, num_ood_samples)
        return FA_X(content=content, styles=styles[:len(batch)]), labels

    def _validate_ood_sample_size(self) -> Tuple[int, int]:
        domain_len = len(self.domains)

        num_domains_to_sample = self.num_domains_to_sample
        num_ood_samples = self.num_ood_samples

        if num_domains_to_sample is None or num_ood_samples is None:
            raise ValueError("Values for collate are empty")

        if num_domains_to_sample >= domain_len:
            raise ValueError(
                f"Cannot sample {num_domains_to_sample} from domain list of length {domain_len}."
            )
        return num_domains_to_sample, num_ood_samples
