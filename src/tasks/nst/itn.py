from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter
from lightning.pytorch.loggers import TensorBoardLogger

from ...models.nst import StyleTransfer_X, StyleTransfer_Y, AdaINModel

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
        gamma: float = 2.0,
        img_log_freq: int = 64,
    ) -> None:
        super().__init__(
            optimizer=optimizer,
            scheduler=scheduler,
        )

        self.network = network

        self.gamma = gamma

        self.loss = self._combined_loss
        self.metrics = {
            'content_loss': self._content_loss,
            'style_loss': self._style_loss,
        }

        self.img_log_freq = img_log_freq

        # hyperparameters: TODO: move to config file
        # self.content_weight = 1 # default value
        # self.style_weight = 20 # default value
        # self.learning_rate = 1e-3 # default value


        # cam
        self.weight_softmax = torch.squeeze(list(self.squeeze_net.parameters())[-2])

        # style image
        # TODO: some preprocessing for self.style / better code for this part
        style = ut.tensor_load_rgbimage(style_image, size=style_size)
        style = ut.preprocess_batch(style.repeat(self.batch_size, 1, 1, 1))
        self.gram_style = [ut.gram_matrix(y) for y in self.vgg16(ut.subtract_imagenet_mean_batch(style))]



    def hook_feature(self, module, input, output):
        self.features_blobs.append(output.data.cpu().numpy())

    def _returnCAM(self, feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        nc, h, w = feature_conv.shape
        cam = torch.matmul(weight_softmax[class_idx],feature_conv.view(nc, h*w))
        return cam

    def training_step(self, batch, batch_idx):

        ### cam loss
        y = self.forward(batch)
        y_cam = ut.subtract_mean_std_batch(ut.preprocess_batch(y))

        x = batch.clone()
        x_cam = ut.subtract_mean_std_batch(ut.preprocess_batch(x))

        logit_x = self.squeeze_net(x_cam); logit_y = self.squeeze_net(y_cam)


        cam_loss = 0
        label = []

        ## cam loss and category loss
        for i in range(len(x_cam)):
            h_x  = F.softmax(logit_x[i])#.data.squeeze()
            probs_x, idx_x = h_x.sort(0, True)

            h_y  = F.softmax(logit_y[i])#.data.squeeze()
            probs_y, idx_y = h_y.sort(0, True)

            x_cam = self._returnCAM(self.features_blobs[0][i], self.weight_softmax, idx_x[0])
            y_cam = self._returnCAM(self.features_blobs[0][i], self.weight_softmax, idx_y[0])

            cam_loss += F.mse_loss(x_cam, y_cam)
            label.append(idx_x[0])

        cam_loss *= 80
        category_loss = 10000 * F.cross_entropy(logit_y, torch.LongTensor(label).to(self.device))



        ### content loss
        x = ut.subtract_imagenet_mean_batch(x)
        y = ut.subtract_imagenet_mean_batch(y)
        features_x = self.vgg16(x)
        features_y = self.vgg16(y)

        content_loss = self.content_weight * F.mse_loss(features_x[2].data, features_y[2])


        ### style loss
        style_loss = 0
        for m in range(len(features_x)):
            gram_y = ut.gram_matrix(features_y[m])
            style_loss += self.style_weight* F.mse_loss(gram_y, self.gram_style[m].data[:self.batch_size*batch_idx, :, :])


        ### total loss
        total_loss = cam_loss + category_loss + content_loss + style_loss
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
