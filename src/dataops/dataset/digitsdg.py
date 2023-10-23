from __future__ import annotations


from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from image import create_image_loader
from typing import List
from ..dataset.base import DatasetConfig, DatasetPartition, ImageDataset, ImageReader

__all__ = ["DigitsDGDataset"]


class DigitsDGDataset(ImageDataset):
    data_url: str = "https://drive.google.com/u/0/uc?id=15V7EsHfCcfbKgsDmzQKj_DfXt_XYp_P7&export=download"
    dataset_name = "DigitsDG"

    def __init__(self, config: DatasetConfig, partition: DatasetPartition) -> None:
        super().__init__(config, partition)

        if partition is DatasetPartition.TEST:
            raise ValueError("Test dataset is not supported")

    def _fetch_data(self) -> Mapping[str, List[ImageReader]]:
        data_root_path = self.config.dataset_path_root
        domain_names = self.config.domains

        reference_label_map = defaultdict(list)

        for domain in domain_names:
            dataset_path = Path(f"{data_root_path}/{domain}/{self.partition.value}/")

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
