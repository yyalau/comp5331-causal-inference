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

    def _log_images(self, writer: SummaryWriter, eval_output: ClassificationEvalOutput[ERM_X], *, prefix: str) -> None:
        eval_output_x = eval_output.x.detach().cpu()
        eval_output_y = eval_output.y.detach().cpu()
        eval_output_y_hat = eval_output.y_hat.detach().cpu()
        batch_size, num_classes = eval_output_y_hat.shape

        nrows = batch_size
        ncols = 2
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            sharex='col', sharey='col',
            squeeze=False,
            # The size needs to be large enough to avoid overlapping text in the class probability vector
            figsize=((ncols * 12), (nrows * 12)),
        )

        grid_idx = 1
        for row in range(nrows):
            example_input_img = torch.einsum('chw->hwc', eval_output_x[row])
            example_gt_class_idx = eval_output_y[row].argmax().item()
            example_class_prob = torch.softmax(eval_output_y_hat[row].unsqueeze(1), dim=0)
            example_pred_class = example_class_prob.argmax().item()

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
                    ax.imshow(example_class_prob, aspect='auto')

                    for c_idx in list(range(num_classes)):
                        if c_idx == example_pred_class:
                            color = 'g' if c_idx == example_gt_class_idx else 'r'
                        else:
                            color = 'y' if c_idx == example_gt_class_idx else 'w'

                        ax.text(0, c_idx, f'{example_class_prob[c_idx, 0].item():.3f}', ha='center', va='center', color=color)

                grid_idx += 1

        fig.tight_layout()

        writer.add_figure(f'images/{prefix}batch', fig, global_step=self.global_step)
