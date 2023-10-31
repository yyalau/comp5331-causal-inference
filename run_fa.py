from __future__ import annotations

from pathlib import Path

from lightning.pytorch.cli import LightningCLI

from src.datamodules.classification import FADataModule
from src.tasks.classification import FATask

EXPERIMENTS_DIR = Path(__file__).parent / 'experiments'

def cli():
    LightningCLI(
        model_class=FATask,
        datamodule_class=FADataModule,
        trainer_defaults={
            'logger': {
                'class_path': 'TensorBoardLogger',
                'init_args': {
                    'save_dir': EXPERIMENTS_DIR,
                    'name': 'fa',
                },
            },
        },
        auto_configure_optimizers=False,
    )

if __name__ == '__main__':
    cli()
