import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import lightning as L

from .base import StyleTransfer_X, StyleTransfer_Y, StyleTransferModel
from .modules import TransformerNet
from . import utils as ut

__all__ = ['ItnModel']


class ItnModel(nn.Module, StyleTransferModel):
    """
    Represents an Image Transfer Network [1]_ style transfer model for images.

    References
    ----------
    .. [1] Yijun Liu, Zuoteng Xu, Wujian Ye, Ziwen Zhang, Shaowei Weng, Chin-Chen Chang,
       and Huajin Tang. 2019. Image Neural Style Transfer With Preserving the Salient Regions.
       *IEEE Access 7* (2019), 40027--40037. <https://doi.org/10.1109/access.2019.2891576>
    """
    def __init__(self):
        super().__init__()

        # localization network
        self.features_blobs = []

        self.squeeze_net = models.squeezenet1_1(pretrained=True)
        self.squeeze_net.eval()
        self.squeeze_net._modules.get('features')[1].register_forward_hook(self.hook_feature)

        # cam
        self.weight_softmax = torch.squeeze(list(self.squeeze_net.parameters())[-2])

        # perception network
        self.vgg16 = models.vgg16(pretrained=True)

        # main model: image transfer network
        self.transfer_model = TransformerNet()

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

    def forward(self, x: StyleTransfer_X) -> StyleTransfer_Y:
        y = self.transfer_model(x)
        return y

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


if __name__ == "__main__":

    # TODO: test compatability with data loader
    # TODO: not yet tested if the code works / model is training properly
    model = ImageTransferNetwork()
