from __future__ import annotations

from pathlib import Path

from lightning.pytorch.cli import LightningCLI

# from src.dataops.datamodules import ViTDataModule
from src.modelops.tasks import ViTTask

def cli():
    experiment_name = 'vit'
    checkpoint_dir = Path(__file__).parent / 'data' / experiment_name

    LightningCLI(
        model_class=ViTTask,
        # datamodule_class=ViTDataModule,
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
