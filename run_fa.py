from __future__ import annotations

from pathlib import Path

from lightning.pytorch.cli import LightningCLI

from src.datamodules.classification import FADataModule
from src.tasks.classification import FATask

def cli():
    experiment_name = 'fa'
    checkpoint_dir = Path(__file__).parent / 'data' / experiment_name

    LightningCLI(
        model_class=FATask,
        datamodule_class=FADataModule,
        trainer_defaults={
            'logger': {
                'class_path': 'TensorBoardLogger',
                'init_args': {
                    'save_dir': checkpoint_dir,
                    'name': experiment_name,
                },
            },
        },
        auto_configure_optimizers=False,
    )

if __name__ == '__main__':
    cli()
