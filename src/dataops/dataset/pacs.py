from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import List

from ..dataset.base import DatasetConfig, DatasetPartition, ImageDataset, ImageReader, SupportedDatasets
from ..image import ImageLoader


__all__ = ["PACSDataset", 'PACSDomains']

class PACSDomains(str, Enum):
    ART = 'art_painting'
    CARTOON = 'cartoon'
    PHOTO = 'photo'
    SKETCH = 'sketch'

class PACSDataset(ImageDataset):
    data_url: str = "https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE"
    dataset_name: str = SupportedDatasets.PACS.value

    @classmethod
    def validate_domains(cls, domains: List[str]) -> None:
        for name in domains:
            try:
                PACSDomains(name)
            except ValueError:
                raise ValueError(f'Domain `{name}` is not valid for {cls.dataset_name}.')

    def __init__(self, config: DatasetConfig, partition: DatasetPartition) -> None:
        super().__init__(config, partition)

    def _fetch_data(self) -> Mapping[str, List[ImageReader]]:
        data_root_path = self.dataset_path_root
        domain_labels = self.domains

        referance_label_map = defaultdict(list)

        for domain_name in domain_labels:
            files_to_search = self._get_file_name(domain_name)

            for file_name in files_to_search:
                file_path = Path(f'{data_root_path}/splits/{file_name}')

                with open(file_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        path, label = line.strip().split(" ")
                        path = Path(f'{data_root_path}/images/{path}')
                        label = int(label)
                        image_loader = ImageLoader(
                            path, self.lazy, self.preprocessor_params
                        )
                        image_data_reader = ImageReader(image_loader, label)
                        referance_label_map[domain_name].append(image_data_reader)

        return referance_label_map


    def _get_file_name(self, domain_name: str) -> list[str]:
        file_names = ["_".join([domain_name, "train_kfold.txt"])]
        if self.partition is DatasetPartition.VALIDATE:
            file_names = ["_".join([domain_name, "crossval_kfold.txt"])]
        elif self.partition is DatasetPartition.TEST:
            file_names = ["_".join([domain_name, "test_kfold.txt"])]
        elif self.partition is DatasetPartition.ALL:
            file_names.append("_".join([domain_name, "test_kfold.txt"]))
            file_names.append("_".join([domain_name, "crossval_kfold.txt"]))

        return file_names
