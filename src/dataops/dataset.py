from __future__ import annotations

from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence, MutableSequence
import copy
from dataclasses import dataclass
from enum import Enum
import func
import image
import numpy as np
import numpy.typing as npt
import os
from PIL import Image
from pathlib import Path

from torch.utils.data import Dataset
import torch
from typing import Self, List, Tuple

from pydantic import BaseModel


from augmentation import RandAugment

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
    lazy: bool
    domains: List[str]
    extension: str
    num_domains_to_sample: int
    num_ood_samples: int
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

    def collate_fn(
        self,
        batch: Sequence[DatasetOutput]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate and process a batch of samples.
        This method returns a data tensor of shape: \n
        `[batch_size, (N*k)+1, height, width, channels]` \n
        and a label tensor of shape: \n
        `[batch_size, (N*k)+1]` \n
        where N is the number of domains that have been sampled, K is the
        number of data objects sampled from the domain and the loaded data
        object which is the first index in both the data tensor and the label
        tensor.
        """

        domain_data_map = copy.deepcopy(self.domain_data_map)
        num_domains_to_sample = self.config.num_domains_to_sample
        num_ood_samples = self.config.num_ood_samples

        batch_data, batch_labels = [], []
        for output in batch:
            image_tensor = output.image_tensor
            image_label = output.label
            image_domain = output.domain
            data, labels = [image_tensor], [image_label]
            ood_domain_list = func.sample_dictionary(
                domain_data_map,
                num_domains_to_sample,
                lambda x: x != image_domain
            )
            for ood_domain in ood_domain_list:
                sample_list = func.sample_sequence_and_remove_from_population(
                    domain_data_map[ood_domain],
                    num_ood_samples
                )
                for sample in sample_list:
                    data.append(torch.from_numpy(sample.load()))
                    labels.append(sample.label)

            batch_data.append(torch.stack(data))
            batch_labels.append(labels)

        return torch.stack(batch_data), torch.from_numpy(np.array(batch_labels))

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
        extension = self.config.extension
        partition = self.partition
        if partition is DatasetPartition.VALIDATE:
            return "_".join([domain_name, "".join(["cross", partition, extension])])
        return "_".join([domain_name, "".join([partition, extension])])
