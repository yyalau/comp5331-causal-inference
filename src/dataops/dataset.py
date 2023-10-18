from __future__ import annotations

from abc import abstractmethod
from augmentation import RandAugment
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence, MutableSequence
from dataclasses import dataclass
from enum import Enum
import func
import image
import numpy as np
import numpy.typing as npt
import os
from PIL import Image
from pathlib import Path
from pydantic import BaseModel

from torch.utils.data import Dataset
import torch
from typing import Self, Tuple


__all__ = ["ImageDataset", "PACSDataset", "DatasetPartition", "DatasetConfig", "DatasetOutput"]


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
    domains: Sequence[str]
    lazy: bool
    rand_augment: Tuple[float, float]


class DatasetPartition(str, Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATE = "val"


@dataclass
class ImageDataLoader:
    load: Callable[[], npt.NDArray[np.float32]]
    label: int


class ImageDataset(Dataset[DatasetOutput]):
    def __init__(
            self,
            config: DatasetConfig,
            partition: DatasetPartition
    ) -> None:
        super().__init__()
        self.config = config
        self.partition = partition
        self.domain_data_map = self._fetch_data()
        self.len = sum(
            len(image_loader)
            for image_loader in self.domain_data_map.values()
        )
        self.transforms = RandAugment(*self.config.rand_augment)

    def __getitem__(self, idx: int) -> DatasetOutput:
        """
        Get the preprocessed item at the specified index.

        Returns:
            do(X) and Y, where `do` is defined in `_preprocess`
        """
        domain, item = func.get_flattened_index(self.domain_data_map, idx)
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
    def _fetch_data(self) -> Mapping[str, MutableSequence[ImageDataLoader]]:
        pass

    def num_domains(self) -> int:
        return len(self.domain_data_map.keys())

    def get_domain_data(self) -> Mapping[str, MutableSequence[ImageDataLoader]]:
        return self.domain_data_map

class PACSDataset(ImageDataset):

    def __init__(
        self,
        config: DatasetConfig,
        partition: DatasetPartition
    ) -> None:
        super().__init__(config, partition)

    @classmethod
    def download(
        cls,
        config: DatasetConfig,
        partition: DatasetPartition
    ) -> Self:
        raise NotImplementedError()

    def _fetch_data(self) -> Mapping[str, MutableSequence[ImageDataLoader]]:
        data_root_path = self.config.data_path
        data_reference_path = self.config.label_path
        domain_labels = self.config.domains

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
                    image_loader = image.create_image_loader(
                        path, self.config.lazy
                    )
                    image_data_loader = ImageDataLoader(image_loader, label)
                    referance_label_map[domain_name].append(image_data_loader)

        return referance_label_map

    def _get_file_name(self, domain_name: str) -> str:
        partition = self.partition
        if partition is DatasetPartition.VALIDATE:
            return "_".join([domain_name, "".join(["cross", partition, "_kfold.txt"])])
        return "_".join([domain_name, "".join([partition, "_kfold.txt"])])
