from __future__ import annotations


from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import List
from ..dataset.base import DatasetConfig, DatasetPartition, ImageDataset, ImageReader, SupportedDatasets
from ..image import create_image_loader

__all__ = ["DigitsDGDataset"]


class DigitsDGDataset(ImageDataset):
    data_url: str = "https://drive.google.com/u/0/uc?id=15V7EsHfCcfbKgsDmzQKj_DfXt_XYp_P7&export=download"
    dataset_name = SupportedDatasets.Digits

    def __init__(self, config: DatasetConfig, partition: DatasetPartition) -> None:
        super().__init__(config, partition)

        if partition is DatasetPartition.TEST:
            raise ValueError("Test dataset is not supported")

    def _fetch_data(self) -> Mapping[str, List[ImageReader]]:
        data_root_path = self.dataset_path_root
        domain_names = self.domains

        reference_label_map = defaultdict(list)
        ds_folders: list[Path] = []

        if self.partition is DatasetPartition.ALL:
            for domain in domain_names:
                ds_folders.append(Path(f"{data_root_path}/{domain}/train/"))
                ds_folders.append(Path(f"{data_root_path}/{domain}/val/"))
        else:
            for domain in domain_names:
                ds_folders.append(Path(f"{data_root_path}/{domain}/{self.partition.value}/"))

        for folder in ds_folders:
            if folder.exists() and folder.is_dir():
                for img_path in folder.iterdir():
                    if img_path.is_file():
                        label = int(folder.name)
                        image_loader = create_image_loader(
                            img_path, self.lazy
                        )
                        domain = img_path.parent.parent.name
                        image_data_loader = ImageReader(image_loader, label)
                        reference_label_map[domain].append(image_data_loader)

        return reference_label_map
