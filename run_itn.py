from __future__ import annotations

from pathlib import Path

from lightning.pytorch.cli import LightningCLI

from src.datamodules.nst import AdaINDataModule
from src.tasks.nst import ItnTask

EXPERIMENTS_DIR = Path(__file__).parent / 'experiments'

def cli():
    LightningCLI(
        model_class=ItnTask,
        datamodule_class=AdaINDataModule,
        trainer_defaults={
            'logger': {
                'class_path': 'TensorBoardLogger',
                'init_args': {
                    'save_dir': EXPERIMENTS_DIR,
                    'name': 'itn',
                },
            },
        },
        auto_configure_optimizers=False,
    )

if __name__ == '__main__':
    cli()
