from __future__ import annotations

from collections.abc import Callable, Iterable

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter

from ...models.classification.fa import FA_X, FAModel, FAST, FAFT, FAGT

from .base import ClassificationTask, ClassificationEvalOutput

__all__ = ['FATask', 'FA_X']


class FATask(ClassificationTask[FA_X]):
    __doc__ = ClassificationTask.__doc__

    def __init__(
        self,
        classifier: FAModel,
        *,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], Optimizer],
        scheduler: Callable[[Optimizer], LRScheduler],
        img_log_freq: int = 64,
        img_log_max_examples_per_batch: int = 4,
    ) -> None:
        super().__init__(
            classifier=classifier,
            optimizer=optimizer,
            scheduler=scheduler,
            img_log_freq=img_log_freq,
            img_log_max_examples_per_batch=img_log_max_examples_per_batch,
        )

    def _log_images(self, writer: SummaryWriter, eval_output: ClassificationEvalOutput[FA_X], *, prefix: str) -> None:
        num_examples = min(eval_output.x['content'].shape[0], self.img_log_max_examples_per_batch)

        eval_output_x_content = eval_output.x['content'].detach().cpu().float()[:num_examples]
        eval_output_x_styles = [style.detach().cpu().float()[:num_examples] for style in eval_output.x['styles']]
        eval_output_y = eval_output.y.detach().cpu().float()[:num_examples]
        eval_output_y_hat = eval_output.y_hat.detach().cpu().float()[:num_examples]
        num_classes = eval_output_y_hat.shape[1]
        num_styles = len(eval_output_x_styles)

        x_tildes = self.classifier.get_x_tildes(eval_output.x) if isinstance(self.classifier, (FAST, FAGT)) else []
        x_nsts = [x_tilde.detach().cpu().float()[:num_examples] for x_tilde in x_tildes]
        num_nst_applied = len(x_nsts)

        x_hats = self.classifier.get_x_hats(eval_output.x) if isinstance(self.classifier, (FAFT, FAGT)) else []
        x_fsts = [x_hat.detach().cpu().float()[:num_examples] for x_hat in x_hats]
        num_fst_applied = len(x_fsts)

        nrows = num_examples
        ncols = 1 + num_styles + num_nst_applied + num_fst_applied + 1
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            sharex='col', sharey='col',
            squeeze=False,
            # The size needs to be large enough to avoid overlapping text in the class probability vector
            figsize=((ncols * 12), (nrows * 12)),
        )

        grid_idx = 1
        for row in range(nrows):
            example_input_content = torch.einsum('chw->hwc', eval_output_x_content[row])
            example_input_styles = [torch.einsum('chw->hwc', style[row]) for style in eval_output_x_styles]

            example_x_nsts = [torch.einsum('chw->hwc', x_nst[row]) for x_nst in x_nsts]
            example_x_fsts = [torch.einsum('chw->hwc', x_fst[row]) for x_fst in x_fsts]

            example_gt_class_idx = eval_output_y[row].argmax().item()
            example_class_prob = torch.softmax(eval_output_y_hat[row].unsqueeze(1), dim=0)
            example_pred_class = example_class_prob.argmax().item()

            for col in range(ncols):
                ax: Axes = axes[row, col]

                if row == 0:
                    if col == 0:
                        ax.set_title('Content Image')
                    elif 1 <= col < 1 + num_styles:
                        ax.set_title('Style Image')
                    elif 1 + num_styles <= col < 1 + num_styles + num_nst_applied:
                        ax.set_title('Applied Image (NST)')
                    elif 1 + num_styles + num_nst_applied <= col < 1 + num_styles + num_nst_applied + num_fst_applied:
                        ax.set_title('Applied Image (FST)')
                    elif col == 1 + num_styles + num_nst_applied + num_fst_applied:
                        ax.set_title('Class Probabilities')

                if col == 0:
                    ax.imshow(example_input_content)
                elif 1 <= col < 1 + num_styles:
                    style_idx = col - 1
                    ax.imshow(example_input_styles[style_idx])
                elif 1 + num_styles <= col < 1 + num_styles + num_nst_applied:
                    style_idx = col - 1 - num_styles
                    ax.imshow(example_x_nsts[style_idx])
                elif 1 + num_styles + num_nst_applied <= col < 1 + num_styles + num_nst_applied + num_fst_applied:
                    style_idx = col - 1 - num_styles - num_nst_applied
                    ax.imshow(example_x_fsts[style_idx])
                elif col == 1 + num_styles + num_nst_applied + num_fst_applied:
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
