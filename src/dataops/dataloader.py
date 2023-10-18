from __future__ import annotations

import copy
from dataclasses import dataclass
from dataset import (
    ImageDataset,
    DatasetConfig,
    PACSDataset,
    DatasetPartition,
    DatasetOutput,
)
from enum import Enum
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Tuple, List

from .func import sample_dictionary, sample_sequence_and_remove_from_population



class Dataset(Enum):
    PACS = 1


@dataclass
class DataLoaderConfig:
    batch_size: int
    shuffle: bool
    num_workers: int
    num_domains_to_sample: int
    num_ood_samples: int


class OODDataLoader(DataLoader[DatasetOutput]):
    def __init__(
            self,
            config: DataLoaderConfig,
            dataset: ImageDataset,
    ) -> None:
        self.config = config
        super(OODDataLoader, self).__init__(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            collate_fn=self.collate
        )

    def collate(
        self, batch: List[DatasetOutput]
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
        assert isinstance(self.dataset, ImageDataset)

        domain_data_map = copy.deepcopy(self.dataset.get_domain_data())
        num_domains_to_sample = self.config.num_domains_to_sample
        num_ood_samples = self.config.num_ood_samples

        batch_data, batch_labels = [], []
        for output in batch:
            image_tensor = output.image_tensor
            image_label = output.label
            image_domain = output.domain
            data, labels = [image_tensor], [image_label]
            ood_domain_list = sample_dictionary(
                domain_data_map, num_domains_to_sample, lambda x: x != image_domain
            )
            for ood_domain in ood_domain_list:
                sample_list = sample_sequence_and_remove_from_population(
                    domain_data_map[ood_domain], num_ood_samples
                )
                for sample in sample_list:
                    data.append(torch.from_numpy(sample.load()))
                    labels.append(sample.label)

            batch_data.append(torch.stack(data))
            batch_labels.append(labels)
        return torch.stack(batch_data), torch.from_numpy(np.array(batch_labels))


def create_data_loaders(
    dataloader_config: DataLoaderConfig,
    dataset_config: DatasetConfig,
    dataset: Dataset
) -> Tuple[OODDataLoader, ...]:

    return tuple(
        OODDataLoader(dataloader_config, dataset)
        for dataset in create_dataset(dataset_config, dataset)
    )


def create_dataset(
    dataset_config: DatasetConfig, dataset: Dataset
) -> Tuple[ImageDataset, ...]:
    if dataset is Dataset.PACS:
        return tuple(
            PACSDataset(dataset_config, partition)
            for partition in DatasetPartition
        )

    raise NotImplementedError()


# config = DatasetConfig(
#     data_path="../../data/pacs/pacs_data",
#     label_path="../../data/pacs/Train val splits and h5py files pre-read",
#     domains=["art_painting", "cartoon", "photo"],
#     lazy=True,
#     rand_augment=(10, 10),
# )

# loader_config = DataLoaderConfig(
#     batch_size=10,
#     shuffle=True,
#     num_workers=4,
#     num_domains_to_sample=1,
#     num_ood_samples=10
# )

# train, test, val = create_data_loaders(loader_config, config, Dataset.PACS)


# for i, (X, Y) in enumerate(train):
#     print(i)
