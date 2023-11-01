from __future__ import annotations

from collections.abc import Callable, Iterable

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter

from ...models.classification import ERM_X, ERMModel

from .base import ClassificationTask, ClassificationEvalOutput

__all__ = ['ERMTask', 'ERM_X']


class ERMTask(ClassificationTask[ERM_X]):
    __doc__ = ClassificationTask.__doc__

    def __init__(
        self,
        classifier: ERMModel,
        *,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], Optimizer],
        scheduler: Callable[[Optimizer], LRScheduler],
        img_log_freq: int = 64,
    ) -> None:
        super().__init__(
            classifier=classifier,
            optimizer=optimizer,
            scheduler=scheduler,
            img_log_freq=img_log_freq,
        )

    def _log_images(self, writer: SummaryWriter, eval_output: ClassificationEvalOutput[ERM_X], *, prefix: str, batch_idx: int) -> None:
        batch_size, num_classes = eval_output.y.shape

        nrows = batch_size
        ncols = 2
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex='col', sharey='col', squeeze=False)

        grid_idx = 1
        for row in range(nrows):
            example_input_img = torch.einsum('chw->hwc', eval_output.x[row]).cpu()
            example_class_prob = eval_output.y[row].unsqueeze(1).cpu()

            for col in range(ncols):
                ax: Axes = axes[row, col]

                if row == 0:
                    if col == 0:
                        ax.set_title('Input Image')
                    elif col == 1:
                        ax.set_title('Class Probabilities')

                if col == 0:
                    ax.imshow(example_input_img)
                elif col == 1:
                    ax.imshow(example_class_prob)

                    ax.set_xticks([0], [''])
                    ax.set_yticks(list(range(num_classes)), labels=[f'Class {c_idx}' for c_idx in range(num_classes)])
                    for c_idx in list(range(num_classes)):
                        ax.text(0, c_idx, f'{example_class_prob[c_idx, 0].item():.3f}', ha='center', va='center', color='w')

                grid_idx += 1

        writer.add_figure(f'images/{prefix}batch_{batch_idx}', fig)
