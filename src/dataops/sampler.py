from __future__ import annotations

from torch.utils.data import DistributedSampler
from .dataset.base import ImageDataset
from typing import Iterator, List, Mapping, Optional
from .func import get_flattend_indices_from_key
import torch

__all__ = [
    "DomainBatchSampler"
]

class DomainBatchSampler(DistributedSampler[List[int]]):

    def __init__(
        self,
        image_dataset: ImageDataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        batch_size: int = 1
    ) -> None:
        super().__init__(dataset=image_dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.image_dataset = image_dataset

    def set_dataset(self, image_dataset: ImageDataset) -> None:
        self.image_dataset = image_dataset

    def get_domain_indices(self, image_dataset: ImageDataset) -> Mapping[str, List[int]]:
        domain_data_map = image_dataset.domain_data_map
        res = dict()
        for domain, value in domain_data_map.items():
            indices = list(range(len(value)))
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(indices), generator=g).tolist()  
            res[domain] = indices
        return res

    def __iter__(self) -> Iterator[List[int]]:
        batch_size = self.batch_size
        domain_index_map = self.get_domain_indices(self.image_dataset)
        current_index = 0
        domains_exhausted = set()
        while True:
            if len(domains_exhausted) == len(domain_index_map.keys()):
                break

            for domain in domain_index_map.keys():
                domain_indices = domain_index_map[domain]
                if current_index + batch_size > len(domain_indices):
                    domains_exhausted.add(domain)
                    continue
                yield get_flattend_indices_from_key(
                    domain_index_map,
                    domain,
                    domain_indices[current_index:current_index + batch_size]
                )

            current_index = current_index + batch_size

    def __len__(self) -> int:

        return sum([
            len(value) // self.batch_size
            for value in self.get_domain_indices(self.image_dataset).values()
        ])
