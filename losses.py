"""Loss functions"""
from typing import Sequence, Optional, Tuple, Callable, Union

import numpy as np
import torch

from torch import nn
from torch.nn import functional as F

from keras import backend as K

# Loss Function and coefficients to be used during training:
def dice_coefficient(y_true, y_pred):
    smoothing_factor = 1
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    return (2. * K.sum(flat_y_true * flat_y_pred) + smoothing_factor) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smoothing_factor)

def dice_coefficient_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

######################################################################

class CombinedLoss(torch.nn.Module):
    """Defines a loss function as a weighted sum of combinable loss criteria.
    Args:
        criteria: List of loss criterion modules that should be combined.
        weight: Weight assigned to the individual loss criteria (in the same
            order as ``criteria``).
        device: The device on which the loss should be computed. This needs
            to be set to the device that the loss arguments are allocated on.
    """
    def __init__(
            self,
            criteria: Sequence[torch.nn.Module],
            weight: Optional[Sequence[float]] = None,
            device: torch.device = None
    ):
        super().__init__()
        self.criteria = torch.nn.ModuleList(criteria)
        self.device = device
        if weight is None:
            weight = torch.ones(len(criteria))
        else:
            weight = torch.as_tensor(weight, dtype=torch.float32)
            assert weight.shape == (len(criteria),)
        self.register_buffer('weight', weight.to(self.device))

    def forward(self, *args):
        loss = torch.tensor(0., device=self.device)
        for crit, weight in zip(self.criteria, self.weight):
            loss += weight * crit(*args)
        return loss

def _channelwise_sum(x: torch.Tensor):
    """Sum-reduce all dimensions of a tensor except dimension 1 (C)"""
    reduce_dims = tuple([0] + list(range(x.dim()))[2:])  # = (0, 2, 3, ...)
    return x.sum(dim=reduce_dims)


# TODO: Dense weight support
def dice_loss(probs, target, weight=1., eps=0.0001, smooth=0.):
    # Probs need to be softmax probabilities, not raw network outputs
    tsh, psh = target.shape, probs.shape

    if tsh == psh:  # Already one-hot
        onehot_target = target.to(probs.dtype)
    elif tsh[0] == psh[0] and tsh[1:] == psh[2:]:  # Assume dense target storage, convert to one-hot
        onehot_target = torch.zeros_like(probs)
        onehot_target.scatter_(1, target.unsqueeze(1), 1)
    else:
        raise ValueError(
            f'Target shape {target.shape} is not compatible with output shape {probs.shape}.'
        )
    # if weight is None:
    #     weight = torch.ones(probs.shape[0], dtype=probs.dtype)  # (C,)
    # if ignore_index is not None:
    #     weight[:, ignore_index] = 0.

    intersection = probs * onehot_target  # (N, C, ...)
    numerator = 2 * _channelwise_sum(intersection) + smooth  # (C,)
    denominator = probs + onehot_target  # (N, C, ...)
    denominator = _channelwise_sum(denominator) + smooth + eps  # (C,)
    loss_per_channel = 1 - (numerator / denominator)  # (C,)
    weighted_loss_per_channel = weight * loss_per_channel  # (C,)
    return weighted_loss_per_channel.mean()  # ()


class DiceLoss(torch.nn.Module):
    """Generalized Dice Loss, as described in https://arxiv.org/abs/1707.03237.
    Works for n-dimensional data. Assuming that the ``output`` tensor to be
    compared to the ``target`` has the shape (N, C, D, H, W), the ``target``
    can either have the same shape (N, C, D, H, W) (one-hot encoded) or
    (N, D, H, W) (with dense class indices, as in
    ``torch.nn.CrossEntropyLoss``). If the latter shape is detected, the
    ``target`` is automatically internally converted to a one-hot tensor
    for loss calculation.
    Args:
        apply_softmax: If ``True``, a softmax operation is applied to the
            ``output`` tensor before loss calculation. This is necessary if
            your model does not already apply softmax as the last layer.
            If ``False``, ``output`` is assumed to already contain softmax
            probabilities.
        weight: Weight tensor for class-wise loss rescaling.
            Has to be of shape (C,). If ``None``, classes are weighted equally.
        smooth: Smoothing term that is added to both the numerator and the
            denominator of the dice loss formula.
    """
    def __init__(
            self,
            apply_softmax: bool = True,
            weight: Optional[torch.Tensor] = None,
            smooth: float = 0.
    ):
        super().__init__()
        if apply_softmax:
            self.softmax = torch.nn.Softmax(dim=1)
        else:
            self.softmax = lambda x: x  # Identity (no softmax)
        self.dice = dice_loss
        if weight is None:
            weight = torch.tensor(1.)
        self.register_buffer('weight', weight)
        self.smooth = smooth

    def forward(self, output, target):
        probs = self.softmax(output)
        return self.dice(probs, target, weight=self.weight, smooth=self.smooth)


