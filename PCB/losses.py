import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

######################################################################
# Cross-entropy loss for soft-label
# --------------------------------------------------------------------
class SoftLabelLoss(nn.Module):
    def __init__(self):
        super(SoftLabelLoss, self).__init__()
        self.eps = 1e-9

    def forward(self, input, target, mask=None):
        if input.dim() > 2:  # N defines the number of images, C defines channels,  K class in total
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        maxRow, _ = torch.max(input.data, 1)  # outputs.data  return the index of the biggest value in each row
        maxRow = maxRow.unsqueeze(1)
        input.data = input.data - maxRow
        loss = self.loss_cross_entropy(input, target, mask)
        return loss

    def loss_cross_entropy(self, input_soft, target_soft, mask=None, reduce=True):
        input_soft = F.log_softmax(input_soft, dim=1)
        result = -target_soft * input_soft
        result = torch.sum(result, 1)
        if mask is not None:
            result = result * mask
        if reduce:
            if mask is not None:
                result = torch.mean(result) * (mask.shape[0] / (mask.sum() + self.eps))
            else:
                result = torch.mean(result)
        return result
