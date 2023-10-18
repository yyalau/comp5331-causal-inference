from __future__ import annotations

import random

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


@dataclass(frozen=False)
class DatasetConfig:
    dataset_path_root: Path
    domains: List[str]
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

        self.lazy = config.lazy
        self.partition = partition
        self.config = config

        if not config.dataset_path_root.exists():
            data_path = self.download(config.dataset_path_root.parent.name)
            self.config.dataset_path_root = Path(data_path)

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
    def download(cls, destination: str) -> str:
        print(f"Downloading data from {cls.data_url}")

        file_path = download_from_gdrive(cls.data_url, f'{destination}/{cls.dataset_name}.zip')
        print(f"Extracting files from {file_path}")

        return unzip(file_path)

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
        data_root_path = self.config.dataset_path_root
        domain_labels = self.config.domains

        referance_label_map = defaultdict(list)

        for domain_name in domain_labels:
            file_name = self._get_file_name(domain_name)
            file_path = Path(f'{data_root_path}/splits/{file_name}')
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

        if partition is DatasetPartition.TEST:
            raise ValueError('Test dataset is not supported')

    def _fetch_data(self) -> Mapping[str, List[ImageReader]]:
        data_root_path = self.config.dataset_path_root
        domain_names = self.config.domains
        
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

class SplitData(BaseModel):
    train: List[Path]
    test: List[Path]
    val: list[Path]


class OfficeHomeDataset(ImageDataset):
    data_url: str = 'https://drive.google.com/u/0/uc?id=0B81rNlvomiwed0V1YUxQdC1uOTg&export=download&resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw'
    dataset_name = "OfficeHome"

    def __init__(
        self,
        config: DatasetConfig,
        partition: DatasetPartition,
        seed: int = 42,
        train_ratio: float = 0.7,
        test_ratio: float = 0.15,
    ) -> None:
        super().__init__(config, partition)
        self.seed = seed
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.labels = {"Alarm_Clock": 0, "Backpack": 1, "Batteries": 2, "Bed": 3, "Bike": 4, "Bottle": 5, "Bucket": 6, "Calculator": 7, "Calendar": 8, "Candles": 9, "Chair": 10, "Clipboards": 11, "Computer": 12, "Couch": 13, "Curtains": 14, "Desk_Lamp": 15, "Drill": 16, "Eraser": 17, "Exit_Sign": 18, "Fan": 19, "File_Cabinet": 20, "Flipflops": 21, "Flowers": 22, "Folder": 23, "Fork": 24, "Glasses": 25, "Hammer": 26, "Helmet": 27, "Kettle": 28, "Keyboard": 29, "Knives": 30, "Lamp_Shade": 31, "Laptop": 32, "Marker": 33, "Monitor": 34, "Mop": 35, "Mouse": 36, "Mug": 37, "Notebook": 38, "Oven": 39, "Pan": 40, "Paper_Clip": 41, "Pen": 42, "Pencil": 43, "Postit_Notes": 44, "Printer": 45, "Push_Pin": 46, "Radio": 47, "Refrigerator": 48, "Ruler": 49, "Scissors": 50, "Screwdriver": 51, "Shelf": 52, "Sink": 53, "Sneakers": 54, "Soda": 55, "Speaker": 56, "Spoon": 57, "TV": 58, "Table": 59, "Telephone": 60, "ToothBrush": 61, "Toys": 62, "Trash_Can": 63, "Webcam": 64}

    def _fetch_data(self) -> Mapping[str, List[ImageReader]]:
        split_data = self._split_data()

        data_to_fetch = split_data.train

        if self.partition is DatasetPartition.TEST:
            data_to_fetch = split_data.test
        elif self.partition is DatasetPartition.VALIDATE:
            data_to_fetch = split_data.val

        reference_label_map = defaultdict()

        for image_path in data_to_fetch:
            label = self.labels[image_path.parent.name]
            domain = image_path.parent.parent.name 
            image_loader = create_image_loader(
                image_path.as_uri(), self.config.lazy
            )

            image_data_loader = ImageReader(image_loader, label)
            reference_label_map[domain].append(image_data_loader)

        return reference_label_map

    def _split_data(self) -> SplitData:
        data_path = self.config.dataset_path_root
        domains = self.config.domains
        random.seed(self.seed)

        image_paths: list[Path] = []

        for domain in domains:
            domain_path = Path(f'{data_path}/{domain}/')

            # Iterate over each image
            for class_dir in domain_path.iterdir():
                if class_dir.is_dir():
                    for image_path in class_dir.iterdir():
                        image_paths.append(image_path)
        
        random.shuffle(image_paths)

        num_images = len(image_paths)
        num_train = int(num_images * self.train_ratio)
        num_test = int(num_images * self.test_ratio)

        train = image_paths[:num_train]
        test = image_paths[num_train:num_train + num_test]
        val = image_paths[num_train + num_test:]

        return SplitData(train=train, test=test, val=val)

