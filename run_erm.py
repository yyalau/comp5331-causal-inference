from __future__ import annotations

from pathlib import Path

from lightning.pytorch.cli import LightningCLI

from src.datamodules.classification import ERMDataModule
from src.tasks.classification import ERMTask

def cli():
    experiment_name = 'erm'
    checkpoint_dir = Path(__file__).parent / 'data' / experiment_name

    LightningCLI(
        model_class=ERMTask,
        datamodule_class=ERMDataModule,
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
