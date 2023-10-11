from __future__ import annotations

from lightning.pytorch.cli import LightningCLI

# from src.dataops.datamodules import ERMDataModule
from src.modelops.tasks import ERMTask

def cli():
    LightningCLI(
        model_class=ERMTask,
        # datamodule_class=ERMDataModule,
        auto_configure_optimizers=False,
    )


if __name__ == '__main__':
    cli()
