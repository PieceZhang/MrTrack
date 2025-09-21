# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from siamban.core.config import cfg
from siamban.models.iou_loss import linear_iou


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    if cfg.BAN.BAN:
        diff = (pred_loc - label_loc).abs()
        diff = diff.sum(dim=1)
    else:
        diff = None
    loss = diff * loss_weight
    return loss.sum().div(pred_loc.size()[0])


def select_iou_loss(pred_loc, label_loc, label_cls):
    label_cls = label_cls.reshape(-1)
    pos = label_cls.data.eq(1).nonzero().squeeze().cuda()

    pred_loc = pred_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    pred_loc = torch.index_select(pred_loc, 0, pos)

    label_loc = label_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    label_loc = torch.index_select(label_loc, 0, pos)

    return linear_iou(pred_loc, label_loc)


def invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    Invariance loss: Ensures that the representations from two different augmentations are consistent.
    We use Mean Squared Error (MSE) to compute the difference between the two views.
    """
    return F.mse_loss(z1, z2)


def variance_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Variance loss: Computes the standard deviation for each feature dimension across the batch.
    If the standard deviation is below 1, it applies a penalty using ReLU.

    This prevents collapse by encouraging each dimension to have sufficient variance.
    """
    # Compute the standard deviation for each dimension, adding eps for numerical stability
    std = torch.sqrt(z.var(dim=0) + eps)
    # Penalize if the standard deviation is less than 1
    loss = torch.mean(F.relu(1 - std))
    return loss


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    """
    Covariance loss: Encourages independence between feature dimensions.

    The process is as follows:
      1. Center the features by subtracting the mean.
      2. Compute the covariance matrix of the centered features.
      3. Zero out the diagonal elements (self-correlation) of the covariance matrix.
      4. Compute the sum of squared off-diagonal elements and normalize by the feature dimension.
    """
    N, D = z.size()
    # Center the features by subtracting the mean of each dimension
    z = z - z.mean(dim=0)
    # Compute the covariance matrix (using N-1 for an unbiased estimate)
    cov = (z.T @ z) / (N - 1)
    # Extract the diagonal of the covariance matrix
    diag = torch.diag(cov)
    # Zero out the diagonal to discard self-correlation terms
    cov_without_diag = cov - torch.diag(diag)
    # Compute the loss as the sum of squared off-diagonal elements, normalized by feature dimension
    loss = (cov_without_diag ** 2).sum() / D
    return loss


# def vicreg_loss(
#         z1: torch.Tensor,
#         z2: torch.Tensor,
#         sim_coeff: float = 25.0,
#         std_coeff: float = 25.0,
#         cov_coeff: float = 1.0
# ) -> tuple:
#     """
#     Computes the complete VICReg loss:
#
#       Total Loss = sim_coeff * invariance_loss +
#                    std_coeff * (variance_loss(z1) + variance_loss(z2)) +
#                    cov_coeff * (covariance_loss(z1) + covariance_loss(z2))
#
#     Parameters:
#       z1, z2: Feature representations from two different views/augmentations (shape: [batch_size, feature_dim]).
#       sim_coeff: Weight for the invariance (similarity) loss.
#       std_coeff: Weight for the variance loss.
#       cov_coeff: Weight for the covariance loss.
#
#     Returns:
#       A tuple (total_loss, sim_loss, var_loss, cov_loss) for further inspection.
#     """
#     sim_loss = invariance_loss(z1, z2)
#     var_loss = variance_loss(z1) + variance_loss(z2)
#     cov_loss = covariance_loss(z1) + covariance_loss(z2)
#
#     loss = sim_coeff * sim_loss + std_coeff * var_loss + cov_coeff * cov_loss
#     return loss, sim_loss, var_loss, cov_loss

def vicreg_loss(
        z: torch.Tensor,
        std_coeff: float = 0.06,  # 0.08
        cov_coeff: float = 0.01  # 0.2
) -> tuple:
    """
    https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
    Computes the complete VICReg loss:

      Total Loss = sim_coeff * invariance_loss +
                   std_coeff * (variance_loss(z1) + variance_loss(z2)) +
                   cov_coeff * (covariance_loss(z1) + covariance_loss(z2))

    Parameters:
      z1, z2: Feature representations from two different views/augmentations (shape: [batch_size, feature_dim]).
      sim_coeff: Weight for the invariance (similarity) loss.
      std_coeff: Weight for the variance loss.
      cov_coeff: Weight for the covariance loss.

    Returns:
      A tuple (total_loss, sim_loss, var_loss, cov_loss) for further inspection.
    """

    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    z = z.flatten(-2)  # z = z.mean(-1) # z = z.flatten(-2)
    # var_loss = variance_loss(z)
    # cov_loss = covariance_loss(z)

    std_z = torch.sqrt(z.var(dim=0) + 0.0001)
    var_loss = torch.mean(F.relu(1 - std_z))

    # batch-level contrastive covariance loss
    z = z.transpose(0, 1)  # the cov matrix then becomes B*B, cov_loss then encourages the off-diagonal elements to be zero, decorrelating different samples
    N, D = z.size()
    cov_z = (z.T @ z) / (N - 1)
    cov_loss = off_diagonal(cov_z).pow_(2).sum().div(D)

    loss = std_coeff * var_loss + cov_coeff * cov_loss
    return loss, std_coeff * var_loss, cov_coeff * cov_loss


# Example usage:
if __name__ == "__main__":
    # Assume a batch size of 32 and feature dimension of 128 for demonstration
    batch_size = 32
    feature_dim = 128
    L = 10

    # Simulate feature representations from two different augmentations
    z1 = torch.randn(batch_size, feature_dim, L, requires_grad=True)
    z2 = torch.randn(batch_size, feature_dim, L, requires_grad=True)

    total_loss, var_loss_val, cov_loss_val = vicreg_loss(z1)

    print("Total Loss:", total_loss.item())
    print("Variance Loss:", var_loss_val.item())
    print("Covariance Loss:", cov_loss_val.item())

    # Backward pass: compute gradients
    total_loss.backward()
