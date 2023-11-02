from __future__ import annotations

from pathlib import Path

from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger

from src.datamodules.nst import AdaINDataModule
from src.tasks.nst import AdaINTask

EXPERIMENTS_DIR = Path(__file__).parent / 'experiments'

def cli():
    LightningCLI(
        model_class=AdaINTask,
        datamodule_class=AdaINDataModule,
        trainer_defaults={
            'logger': TensorBoardLogger(save_dir=EXPERIMENTS_DIR, name='nst'),
            'callbacks': [RichProgressBar()],
        },
        auto_configure_optimizers=False,
    )

if __name__ == '__main__':
    cli()
