from __future__ import annotations

from dataclasses import dataclass
from dataops.dataset import (
    DatasetConfig,
    DatasetPartition,
    PACSDataset
)
from enum import Enum
from typing import Optional
from torch.utils.data import DataLoader

class Dataset(Enum):
    PACS = 1


@dataclass
class DataLoaderConfig:
    batch_size: int
    shuffle: bool
    num_workers: int
    num_domains_to_sample: Optional[int]
    num_ood_samples: Optional[int]


# def create_data_loader(
#     dataloader_config: DataLoaderConfig,
#     dataset_config: DatasetConfig,
#     dataset: Dataset
# ) -> Tuple[OODDataLoader, ...]:
#     # return tuple(
#     #     OODDataLoader(dataloader_config, dataset)
#     #     for dataset in create_dataset(dataset_config, dataset)
#     # )


# def create_dataset(
#     dataset_config: DatasetConfig,
#     dataset: Dataset,
#     datasetPartition: DatasetPartition
# ) -> Tuple[ImageDataset, ...]:
#     if dataset is Dataset.PACS:
#         return tuple(
#             PACSDataset(dataset_config, partition) for partition in DatasetPartition
#         )

#     raise NotImplementedError()


config = DatasetConfig(
    dataset_path_root="../../data/pacs",
    train_val_domains=["art_painting", "cartoon", "photo"],
    test_domains=["sketch", "art_painting", "cartoon"],
    lazy=False,
    rand_augment=(10, 10),
)

loader_config = DataLoaderConfig(
    batch_size=10,
    shuffle=True,
    num_workers=4,
    num_domains_to_sample=1,
    num_ood_samples=10,
)

# train, test, val = create_data_loaders(loader_config, config, Dataset.PACS)
if __name__ == '__main__':

    train_dataset = PACSDataset(config, DatasetPartition.TRAIN)
    def collate(batch): train_dataset.multi_ood_collate(batch, 2, 1)


    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate)

    for i, (X, Y) in enumerate(train_loader):
        print(i)
