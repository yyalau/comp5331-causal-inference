from __future__ import annotations


from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from pydantic import BaseModel
import random
from typing import List
from ..dataset.base import DatasetConfig, DatasetPartition, ImageDataset, ImageReader
from ..image import create_image_loader

__all__ = ["OfficeHomeDataset"]


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
                image_path.as_uri(), self.lazy
            )

            image_data_loader = ImageReader(image_loader, label)
            reference_label_map[domain].append(image_data_loader)

        return reference_label_map

    def _split_data(self) -> SplitData:
        data_path = self.dataset_path_root
        domains = self.domains
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
