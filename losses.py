import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class BoundaryAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary')

    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)

        pred_prob = torch.sigmoid(pred)

        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).to(pred.device)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).to(pred.device)

        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)

        pred_edge = torch.sqrt(
            F.conv2d(pred_prob, sobel_x, padding=1) ** 2 +
            F.conv2d(pred_prob, sobel_y, padding=1) ** 2 + 1e-8
        )

        target_edge = torch.sqrt(
            F.conv2d(target, sobel_x, padding=1) ** 2 +
            F.conv2d(target, sobel_y, padding=1) ** 2 + 1e-8
        )

        boundary_dice = (
            2 * (pred_edge * target_edge).sum() + 1e-6
        ) / (
            pred_edge.sum() + target_edge.sum() + 1e-6
        )

        return dice_loss + 0.4 * (1 - boundary_dice)


def get_loss(model_name):
    if model_name == 'unet_boundary':
        return BoundaryAwareLoss()
    return smp.losses.DiceLoss(mode='binary')