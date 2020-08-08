import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, feature1, feature2, target, size_average=True):
        distances = (feature2 - feature1).pow(2).sum(1)  # squared distances
        losses = target.float() * distances \
                 + (1.0 - target).float() * F.relu(self.margin - (distances + self.eps).sqrt().pow(2))
        return losses.mean() if size_average else losses.sum()
