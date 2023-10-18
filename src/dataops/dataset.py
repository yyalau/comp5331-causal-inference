from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable, Mapping

from torch.utils.data import Dataset
import torch
from typing import Tuple, List


from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt
import os
from PIL import Image
from pathlib import Path
from pydantic import BaseModel

from image import create_image_loader
from func import get_flattened_index
from augmentation import RandAugment
from utils import download_from_gdrive, unzip


__all__ = [
    "ImageDataset", "PACSDataset", "DatasetPartition",
    "DatasetConfig", "DatasetOutput", "OfficeHomeDataset", "DigitsDGDataset",
]

class DatasetOutput(BaseModel):
    image_tensor: torch.Tensor
    label: int
    domain: str

    class Config:
        arbitrary_types_allowed = True


@dataclass(frozen=True)
class DatasetConfig:
    data_path: Path
    label_path: Path
    train_val_domains: List[str]
    test_domains: List[str]
    lazy: bool
    rand_augment: Tuple[float, float]


class DatasetPartition(str, Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATE = "val"


@dataclass
class ImageReader:
    load: Callable[[], npt.NDArray[np.float32]]
    label: int


class ImageDataset(Dataset[DatasetOutput]):
    data_url = ""
    dataset_name = ""

    def __init__(
            self,
            config: DatasetConfig,
            partition: DatasetPartition
    ) -> None:
        super().__init__()
        self.data_path = config.data_path
        self.label_path = config.label_path
        self.lazy = config.lazy
        self.partition = partition
        self.domains = (
            config.test_domains
            if partition is DatasetPartition.TEST
            else config.train_val_domains
        )
        self.domain_data_map = self._fetch_data()
        self.len = sum(
            len(image_loader)
            for image_loader in self.domain_data_map.values()
        )
        self.rand_augment = config.rand_augment
        self.transforms = RandAugment(*self.rand_augment)

    @classmethod
    def download(cls, destination: str) -> None:
        print(f"Downloading data from {cls.data_url}")

        file_path = download_from_gdrive(cls.data_url, f'{destination}/{cls.dataset_name}.zip')
        print("Extracting files in dataset ...")

        unzip(file_path)

    def __getitem__(self, idx: int) -> DatasetOutput:
        """
        Get the preprocessed item at the specified index.

        Returns:
            do(X) and Y, where `do` is defined in `_preprocess`
        """
        domain, item = get_flattened_index(self.domain_data_map, idx)
        processed_image = self._preprocess(item.load())
        image_tensor = torch.from_numpy(processed_image)
        label = item.label
        return DatasetOutput(
            image_tensor=image_tensor,
            label=label,
            domain=domain
        )

    def __len__(self) -> int:
        return self.len

    def _preprocess(
        self,
        X: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:

        image = Image.fromarray(X)
        return np.array(self.transforms(image))

    @abstractmethod
    def _fetch_data(self) -> Mapping[str, List[ImageReader]]:
        pass

    def num_domains(self) -> int:
        return len(self.domain_data_map.keys())

    def get_domain_data(self) -> Mapping[str, List[ImageReader]]:
        return self.domain_data_map

class PACSDataset(ImageDataset):
    data_url: str = 'https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE'
    dataset_name = "PACS"

    def __init__(
        self,
        config: DatasetConfig,
        partition: DatasetPartition
    ) -> None:
        super().__init__(config, partition)


    def _fetch_data(self) -> Mapping[str, List[ImageReader]]:
        data_root_path = self.data_path
        data_reference_path = self.label_path
        domain_labels = self.domains

        referance_label_map = defaultdict(list)

        for domain_name in domain_labels:
            file_name = self._get_file_name(domain_name)
            file_path = os.path.join(data_reference_path, file_name)
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    path, label = line.strip().split(" ")
                    path = os.path.join(data_root_path, path)
                    label = int(label)
                    image_loader = create_image_loader(
                        path, self.lazy
                    )
                    image_data_loader = ImageReader(image_loader, label)
                    referance_label_map[domain_name].append(image_data_loader)

        return referance_label_map

    def _get_file_name(self, domain_name: str) -> str:
        if self.partition is DatasetPartition.VALIDATE:
            extension = "crossval_kfold.txt"
        elif self.partition is DatasetPartition.TEST:
            extension = "test_kfold.txt"
        else:
            extension = "train_kfold.txt"

        return "_".join([domain_name, extension])



class DigitsDGDataset(ImageDataset):
    data_url: str = 'https://drive.google.com/u/0/uc?id=15V7EsHfCcfbKgsDmzQKj_DfXt_XYp_P7&export=download'
    dataset_name = "DigitsDG"

    def __init__(
        self,
        config: DatasetConfig,
        partition: DatasetPartition
    ) -> None:
        super().__init__(config, partition)

        if partition.value == 'test':
            raise ValueError('Test dataset is not supported')

    def _fetch_data(self) -> Mapping[str, List[ImageReader]]:
        data_root_path = self.data_path
        domain_names = self.domains

        reference_label_map = defaultdict(list)

        for domain in domain_names:
            dataset_path = Path(f'{data_root_path}/{domain}/{self.partition.value}/')

            for folder in dataset_path.iterdir():
                if folder.exists() and folder.is_dir():
                    for image in folder.iterdir():
                        if image.is_file():
                            label = int(folder.name)
                            image_loader = create_image_loader(
                                image.as_uri(), self.lazy
                            )
                            image_data_loader = ImageReader(image_loader, label)
                            reference_label_map[domain].append(image_data_loader)

        return reference_label_map


class OfficeHomeDataset(ImageDataset):
    data_url: str = 'https://drive.google.com/u/0/uc?id=0B81rNlvomiwed0V1YUxQdC1uOTg&export=download&resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw'
    dataset_name = "OfficeHome"

    def __init__(
        self,
        config: DatasetConfig,
        partition: DatasetPartition
    ) -> None:
        super().__init__(config, partition)
