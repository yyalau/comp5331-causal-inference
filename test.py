from src.datamodules.classification import ERMDataModule

from src.dataops.dataset import DatasetConfig, SupportedDatasets
from pathlib import Path

config = DatasetConfig(dataset_path_root=Path('/home/tan/maral/fast/comp5331/data/digits_dg'),
                       dataset_name=SupportedDatasets.Digits,
                       test_domains=['mnist_m','syn'],
                       train_val_domains=['mnist', 'svhn'],
                       lazy=False,
                       rand_augment=(0.1, 0.1),
                       num_domains_to_sample=None,
                       num_ood_samples=None)

erm = ERMDataModule(config, 32)

print(erm.train_dataloader())