from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter
from lightning.pytorch.loggers import TensorBoardLogger

from ...models.nst import StyleTransfer_X, StyleTransfer_Y, ItnModel

from ..base import EvalOutput, BaseTask

__all__ = ['ItnTask', 'StyleTransfer_X', 'StyleTransfer_Y']


@dataclass(frozen=True)
class ItnEvalOutput(EvalOutput):
    x: StyleTransfer_X
    lazy_y_hat: Callable[[], StyleTransfer_Y]

class ItnTask(BaseTask[StyleTransfer_X, ItnEvalOutput, StyleTransfer_X, StyleTransfer_Y]):
    def __init__(
        self,
        network: ItnModel,
        *,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], Optimizer],
        scheduler: Callable[[Optimizer], LRScheduler],
        w_style: float = 1.0,
        w_content: float = 20.0,
        img_log_freq: int = 64,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
        )

        self.network = network
        self.w_style = w_style
        self.w_content = w_content

        self.loss_fn = self._combined_loss
        self.metrics = {
            'content_loss': self._content_loss,
            'style_loss': self._style_loss,
            'category_loss': self._category_loss,
            'cam_loss': self._cam_loss,
        }

        self.img_log_freq = img_log_freq

        # cam
        self.w_softmax = torch.squeeze(list(self.network.squeeze_net.parameters())[-2])

    def _content_loss_fn(self, feat_x: StyleTransfer_Y, feat_y: StyleTransfer_Y) -> torch.Tensor:
        return F.mse_loss(feat_x[2], feat_y[2])

    def _content_loss(self, features_xc, features_yc, features_xs, label_id_xc, label_id_yc, logit_yc):
        return self._content_loss_fn(features_xc, features_yc)

    def _style_loss_fn(self, feat_xc, feat_xs: StyleTransfer_Y) -> torch.Tensor:

        def gram_matrix(y):
            (b, ch, h, w) = y.size()
            features = y.view(b, ch, w * h)
            features_t = features.transpose(1, 2)
            gram = features.bmm(features_t) / (ch * h * w)
            return gram

        style_loss = 0
        for x,y in zip(feat_xc, feat_xs):
            style_loss += F.mse_loss(*map( gram_matrix, (x, y)))

        return style_loss

    def _style_loss(self, features_xc, features_yc, features_xs, label_id_xc, label_id_yc, logit_yc):
        return self._style_loss_fn(features_yc, features_xs)

    def _category_loss_fn(self, y_hat: StyleTransfer_Y, y: StyleTransfer_Y) -> torch.Tensor:
        return F.cross_entropy(y_hat, y)

    def _category_loss(self, features_xc, features_yc, features_xs, label_id_xc, label_id_yc, logit_yc):
        return self._category_loss_fn(logit_yc, label_id_xc)

    def _cam_loss_fn(self, label_id_x: StyleTransfer_Y, label_id_y: StyleTransfer_Y) -> torch.Tensor:

        def get_cam(feature_conv, class_idx):
            b, nc, h, w = feature_conv.shape #8, 512, 1, 1
            cam = torch.einsum('...ij,...jk->ik',
                               self.w_softmax[class_idx],
                               feature_conv.view(b, nc, -1))
            return cam

        x_cam = get_cam(self.network.features_blobs[0], label_id_x)
        y_cam = get_cam(self.network.features_blobs[1], label_id_y)

        return F.mse_loss(x_cam, y_cam)
    
    def _cam_loss(self, features_xc, features_yc, features_xs, label_id_xc, label_id_yc, logit_yc):
        return self._cam_loss_fn(label_id_xc, label_id_yc)
    

    def _combined_loss(self, features_xc, features_yc, features_xs, label_id_xc, label_id_yc, logit_yc) -> torch.Tensor:
        content_loss = self._content_loss(features_xc, features_yc, features_xs, label_id_xc, label_id_yc, logit_yc)
        style_loss = self._style_loss(features_xc, features_yc, features_xs, label_id_xc, label_id_yc, logit_yc)
        cam_loss = self._cam_loss(features_xc, features_yc, features_xs, label_id_xc, label_id_yc, logit_yc)
        category_loss = self._category_loss(features_xc, features_yc, features_xs, label_id_xc, label_id_yc, logit_yc)
        

        return 10000*category_loss + 80* cam_loss + self.w_style * style_loss + self.w_content * content_loss

    def _eval_step(self, batch, batch_idx) -> ItnEvalOutput:


        '''
        features: pass through vgg
        logit: pass through squeeze net
        y: pass through transfer model
        x: original batch data
        c: content
        s: style
        '''

        ## TODO: refactor this part
        if self.w_softmax.device != batch['content'].device:
            self.w_softmax = self.w_softmax.to(batch['content'].device)


        def get_rank(logits):
            h = F.softmax(logits, dim=1)
            _, idx = h.sort(dim = 1, descending = True)
            return idx[:,0]


        xc, xs = batch['content'], batch['style']
        yc = self.network.transfer_model(xc) # 8, 3, 32,32

        logit_xc, logit_yc = map(self.network.squeeze_net, (xc, yc)) # 8, 1000
        label_id_xc, label_id_yc = map(get_rank, (logit_xc, logit_yc))


        features_xc, features_yc = map(self.network.vgg16.get_states, (xc, yc)) # 8, 1000
        features_xs = self.network.vgg16.get_states(xs)


        # 8, 256, 32, 32-> 8, 256, 16, 16 -> 8, 256, 8, 8 -> 8, 256, 4, 4
        loss = self.loss_fn(features_xc, features_yc, features_xs, label_id_xc, label_id_yc, logit_yc)
        metrics = {name: metric(features_xc, features_yc, features_xs, label_id_xc, label_id_yc, logit_yc)
                   for name, metric in self.metrics.items()}


        return ItnEvalOutput(loss=loss, metrics=metrics, x=batch, lazy_y_hat=lambda: self.network(batch))


    def forward(self, x: StyleTransfer_X) -> StyleTransfer_Y:
        return self.network(x)


    def _process_images(self, eval_output: AdaINEvalOutput, *, prefix: str, batch_idx: int) -> None:
        if batch_idx % self.img_log_freq:
            return

        if isinstance(self.logger, TensorBoardLogger):
            self._log_images(self.logger.experiment, eval_output, prefix=prefix)
        else:
            raise TypeError('Incorrect type of logger')

    def _log_images(self, writer: SummaryWriter, eval_output: AdaINEvalOutput, *, prefix: str) -> None:
        eval_output_x_content = eval_output.x['content'].detach().cpu().float()
        eval_output_x_style = eval_output.x['style'].detach().cpu().float()
        eval_output_y_hat = eval_output.lazy_y_hat().detach().cpu().float()
        batch_size = eval_output_y_hat.shape[0]

        nrows = batch_size
        ncols = 3
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            sharex='col', sharey='col',
            squeeze=False,
            figsize=((ncols * 4), (nrows * 4)),
        )

        grid_idx = 1
        for row in range(nrows):
            example_input_content = torch.einsum('chw->hwc', eval_output_x_content[row])
            example_input_style = torch.einsum('chw->hwc', eval_output_x_style[row])
            example_output_applied = torch.einsum('chw->hwc', eval_output_y_hat[row])

            for col in range(ncols):
                ax: Axes = axes[row, col]

                if row == 0:
                    if col == 0:
                        ax.set_title('Content Image')
                    elif col == 1:
                        ax.set_title('Style Image')
                    elif col == 2:
                        ax.set_title('Applied Image')

                if col == 0:
                    ax.imshow(example_input_content)
                elif col == 1:
                    ax.imshow(example_input_style)
                elif col == 2:
                    ax.imshow(example_output_applied)

                grid_idx += 1

        fig.tight_layout()

        writer.add_figure(f'images/{prefix}batch', fig, global_step=self.global_step)

        fig.clear()

    def training_step(self, batch: StyleTransfer_X, batch_idx: int) -> dict[str, torch.Tensor]:
        eval_output = self._eval_step(batch, batch_idx)
        self._process_images(eval_output, batch_idx=batch_idx, prefix='train_')
        return self._process_eval_loss_metrics(eval_output, prefix='')

    def validation_step(self, batch: StyleTransfer_X, batch_idx: int) -> dict[str, torch.Tensor]:
        eval_output = self._eval_step(batch, batch_idx)
        self._process_images(eval_output, batch_idx=batch_idx, prefix='val_')
        return self._process_eval_loss_metrics(eval_output, prefix='val_')

    def test_step(self, batch: StyleTransfer_X, batch_idx: int) -> dict[str, torch.Tensor]:
        eval_output = self._eval_step(batch, batch_idx)
        self._process_images(eval_output, batch_idx=batch_idx, prefix='test_')
        return self._process_eval_loss_metrics(eval_output, prefix='test_')
