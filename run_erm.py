from __future__ import annotations

from pathlib import Path

from lightning.pytorch.callbacks import ModelCheckpoint
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
            'callbacks': [
                ModelCheckpoint(
                    filename='{epoch}-{step}-{val_loss:.3f}',
                    monitor='val_loss',
                    save_last=True,
                ),
            ],
        },
        auto_configure_optimizers=False,
    )

if __name__ == '__main__':
    cli()
