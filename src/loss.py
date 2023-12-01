"""
Monocular depth estimation loss module.

Modified from https://github.com/vinceecws/Monodepth/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class Loss(nn.Module):
    """
    Loss module for supervised monocular depth estimation.
    """

    def __init__(self, edge_loss_weight=0.4, ssim_loss_weight=0.7, reg_loss_weight=0.5):
        super().__init__()
        self.edge_loss_weight = edge_loss_weight
        self.ssim_loss_weight = ssim_loss_weight
        self.reg_loss_weight = reg_loss_weight

        self.reg_loss = nn.MSELoss()

    def gradients(self, img):
        """
        Calculates the image gradients in x and y axes.
        """
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]
        dx = img[:, :, :, 1:] - img[:, :, :, :-1]
        dy = F.pad(dy, [0, 0, 1, 0], value=0)
        dx = F.pad(dx, [0, 0, 0, 1], value=0)
        return dx, dy

    def smoothness_loss(self, pred, target):
        """
        Calculates the smoothness loss.
        """
        dx_pred, dy_pred = self.gradients(pred)
        dx_true, dy_true = self.gradients(target)

        # e^(-|x|) weights, gradient negatively exponential to weights
        # average over all pixels in C dimension but supposed to be locally smooth?
        weights_x = torch.exp(-torch.mean(torch.abs(dx_true), 1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(dy_true), 1, keepdim=True))

        smoothness_x = torch.mean(torch.abs(dx_pred * weights_x))
        smoothness_y = torch.mean(torch.abs(dy_pred * weights_y))

        smoothness = smoothness_x + smoothness_y
        return smoothness

    def ssim_loss(self, pred, target):
        """
        Calculates the structural similarity (SSIM) loss.
        """
        return 1 - ssim(pred, target, data_range=1.0, size_average=True)

    def regression_loss(self, pred, target):
        """
        Calculates the direct regression loss.
        """
        return torch.sqrt((self.reg_loss(pred, target) + 1e-6))

    def forward(self, x, target):
        """
        Calculate all losses and do weighted sum.
        """
        smoothness_loss = self.edge_loss_weight * self.smoothness_loss(x, target)
        ssim_loss = self.ssim_loss_weight * self.ssim_loss(x, target)
        regression_loss = self.reg_loss_weight * self.regression_loss(x, target)
        return smoothness_loss + ssim_loss + regression_loss
