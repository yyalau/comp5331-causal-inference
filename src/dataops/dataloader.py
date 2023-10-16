from dataclasses import dataclass
from dataset import ImageDataset, DatasetConfig, PACSDataset, DatasetPartition
from enum import Enum
from torch.utils.data import DataLoader
from typing import TypeVar


T = TypeVar('T')
ImageDataLoader = DataLoader[ImageDataset]

class Dataset(Enum):
    PACS = 1


@dataclass
class DataLoaderConfig:
    batch_size: int
    shuffle: bool

def create_data_loaders(
        dataloader_config: DataLoaderConfig,
        dataset_config: DatasetConfig,
        dataset: Dataset
) -> tuple[ImageDataLoader, ImageDataLoader, ImageDataLoader]:

    batch_size = dataloader_config.batch_size
    shuffle = dataloader_config.shuffle

    train_set, test_set, val_set = create_dataset(dataset_config, dataset)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=shuffle)
    return train_loader, test_loader, val_loader

def create_dataset(dataset_config: DatasetConfig, dataset: Dataset):
    if dataset is Dataset.PACS:
        train = PACSDataset(dataset_config, DatasetPartition.TRAIN)
        test = PACSDataset(dataset_config, DatasetPartition.TEST)
        val = PACSDataset(dataset_config, DatasetPartition.VALIDATE)
        return train, test, val


config = DatasetConfig(
data_path="../../data/pacs/pacs_data",
label_path="../../data/pacs/Train val splits and h5py files pre-read",
lazy=True,
domains=["art_painting", "cartoon", "photo"],
extension="_kfold.txt",
num_domains_to_sample=1,
num_ood_samples=3,
rand_augment=(10, 10)
)

loader_config = DataLoaderConfig(
batch_size=10,
shuffle= True
)

train, test, val = create_data_loaders(loader_config, config, Dataset.PACS)