from __future__ import annotations

from lightning.pytorch.cli import LightningCLI

# from src.dataops.datamodules import ClassificationDataModule
from src.modelops.tasks import ClassificationTask

def cli():
    LightningCLI(
        model_class=ClassificationTask,
        # datamodule_class=ClassificationDataModule,
        auto_configure_optimizers=False,
    )


if __name__ == '__main__':
    cli()
