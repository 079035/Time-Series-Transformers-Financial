# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch
import torch as t  # Keep both for compatibility with existing code
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Reference: https://arxiv.org/abs/1708.02002
    
    Args:
        alpha (float or list): Weighting factor in range (0,1) to balance positive vs negative examples
                               or a list of weights for each class
        gamma (float): Exponent of the modulating factor (1 - p_t)^gamma. Default = 2
        reduction (str): 'none' | 'mean' | 'sum'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: predictions from model (before softmax)
            targets: ground truth labels
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # Gather alpha values for each sample based on target class
                alpha_t = torch.tensor(self.alpha, device=targets.device)[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Weighted Binary Cross Entropy with Logits Loss
    Combines sigmoid and binary cross entropy in a numerically stable way
    
    Args:
        pos_weight (float): Weight for positive class. Higher values penalize false negatives more
    """
    def __init__(self, pos_weight=1.0):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: predictions from model (logits for each class)
            targets: ground truth labels (class indices)
        """
        # For binary classification with 2 output classes, extract positive class logits
        if inputs.shape[1] == 2:
            # Get logits for positive class
            pos_logits = inputs[:, 1]
            neg_logits = inputs[:, 0]
            
            # Calculate log softmax manually for numerical stability
            max_val = torch.max(inputs, dim=1, keepdim=True)[0]
            exp_logits = torch.exp(inputs - max_val)
            sum_exp = exp_logits.sum(dim=1, keepdim=True)
            probs = exp_logits / sum_exp
            
            # Binary targets
            binary_targets = targets.float()
            
            # Calculate weighted BCE loss using probabilities
            pos_loss = -binary_targets * torch.log(probs[:, 1] + 1e-7) * self.pos_weight
            neg_loss = -(1 - binary_targets) * torch.log(probs[:, 0] + 1e-7)
            
            loss = pos_loss + neg_loss
            return loss.mean()
        else:
            raise ValueError("WeightedBCEWithLogitsLoss only supports binary classification (2 classes)")


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for handling class imbalance
    
    Args:
        weight (Tensor): a manual rescaling weight given to each class
    """
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: predictions from model
            targets: ground truth labels
        """
        return F.cross_entropy(inputs, targets, weight=self.weight)
