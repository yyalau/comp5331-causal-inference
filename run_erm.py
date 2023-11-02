from __future__ import annotations

from pathlib import Path

from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger

from src.datamodules.classification import ERMDataModule
from src.tasks.classification import ERMTask

EXPERIMENTS_DIR = Path(__file__).parent / 'experiments'

def cli():
    LightningCLI(
        model_class=ERMTask,
        datamodule_class=ERMDataModule,
        trainer_defaults={
            'logger': TensorBoardLogger(save_dir=EXPERIMENTS_DIR, name='erm'),
        },
        auto_configure_optimizers=False,
    )

if __name__ == '__main__':
    cli()
