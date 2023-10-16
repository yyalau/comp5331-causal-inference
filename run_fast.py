from __future__ import annotations

from pathlib import Path

from lightning.pytorch.cli import LightningCLI

# from src.dataops.datamodules import FASTDataModule
from src.modelops.tasks import FASTTask

def cli():
    experiment_name = 'fast'
    checkpoint_dir = Path(__file__).parent / 'data' / experiment_name

    LightningCLI(
        model_class=FASTTask,
        # datamodule_class=FASTDataModule,
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
