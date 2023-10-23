from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import List
from ..dataset.base import DatasetConfig, DatasetPartition, ImageDataset, ImageReader
from ..image import create_image_loader


__all__ = ["PACSDataset"]


class PACSDataset(ImageDataset):
    data_url: str = "https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE"

    dataset_name = "PACS"

    def __init__(self, config: DatasetConfig, partition: DatasetPartition) -> None:
        super().__init__(config, partition)

    def _fetch_data(self) -> Mapping[str, List[ImageReader]]:
        data_root_path = self.dataset_path_root
        domain_labels = self.domains

        referance_label_map = defaultdict(list)

        for domain_name in domain_labels:

            file_name = self._get_file_name(domain_name)
            file_path = Path(f'{data_root_path}/splits/{file_name}')
            with open(file_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    path, label = line.strip().split(" ")
                    path = Path(f'{data_root_path}/images/{path}')
                    label = int(label)
                    image_loader = create_image_loader(
                        path, self.lazy
                    )
                    image_data_reader = ImageReader(image_loader, label)
                    referance_label_map[domain_name].append(image_data_reader)

        return referance_label_map


    def _get_file_name(self, domain_name: str) -> str:
        if self.partition is DatasetPartition.VALIDATE:
            extension = "crossval_kfold.txt"
        elif self.partition is DatasetPartition.TEST:
            extension = "test_kfold.txt"
        else:
            extension = "train_kfold.txt"

        return "_".join([domain_name, extension])
