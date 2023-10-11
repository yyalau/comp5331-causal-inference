from __future__ import annotations

from lightning.pytorch.cli import LightningCLI

# from src.dataops.datamodules import CausalFDDataModule
from src.modelops.tasks import CausalFDTask

def cli():
    LightningCLI(
        model_class=CausalFDTask,
        # datamodule_class=CausalFDDataModule,
        auto_configure_optimizers=False,
    )


if __name__ == '__main__':
    cli()
