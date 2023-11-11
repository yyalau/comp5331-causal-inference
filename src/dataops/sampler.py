from __future__ import annotations

from torch.utils.data.sampler import BatchSampler
from .dataset.base import ImageDataset
from typing import Iterator, List, Mapping
from random import shuffle
from .func import get_flattend_indices_from_key

__all__ = [
    "DomainBatchSampler"
]

class DomainBatchSampler(BatchSampler):

    def __init__(self, image_dataset: ImageDataset, batch_size: int, drop_last: bool = False) -> None:
        self.image_dataset = image_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def get_domain_indices(self, image_dataset: ImageDataset) -> Mapping[str, List[int]]:
        domain_data_map = image_dataset.domain_data_map
        res = dict()
        for domain, value in domain_data_map.items():
            indices = list(range(len(value)))
            shuffle(indices)
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
