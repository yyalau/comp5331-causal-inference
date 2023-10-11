from __future__ import annotations

from lightning.pytorch.cli import LightningCLI

# from src.dataops.datamodules import FASTDataModule
from src.modelops.tasks import FASTTask

def cli():
    LightningCLI(
        model_class=FASTTask,
        # datamodule_class=FASTDataModule,
        auto_configure_optimizers=False,
    )


if __name__ == '__main__':
    cli()
