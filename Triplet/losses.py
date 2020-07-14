import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SigmoidLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self):
        super(SigmoidLoss, self).__init__()
        self.eps = 1e-9

    def forward(self, predit, target, size_average=True):
        predit = torch.sigmoid(predit)
        losses = -(target.float() * torch.log(predit.squeeze()) + (1.0 - target).float() * torch.log(
            1 - predit.squeeze()))
        # losses = (predit.squeeze() - target.float()).pow(2)
        return losses.mean() if size_average else losses.sum()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class ContrastiveLoss_AS(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss_AS, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, flag, output_c1, output_c2, output_p1, output_p2, size_average=True):
        # #Euclid distance
        distances_c = (output_c1 - output_c2).pow(2).sum(1)
        distances_p = (output_p1 - output_p2).pow(2).sum(1)
        distances_d = (((output_c1 - output_p1).pow(2).sum(1) + self.eps).sqrt() - (
                    (output_c2 - output_p2).pow(2).sum(1) + self.eps).sqrt()).pow(2)
        losses_c_same = distances_c
        losses_c_diff = F.relu(self.margin - (distances_c + self.eps).sqrt()).pow(2)
        losses_c = flag * losses_c_same + (1 - flag) * losses_c_diff
        # #Cosine distance
        # distances_c = 1.0 - torch.mul(output_c1, output_c2).sum(1)
        # distances_p = 1.0 - torch.mul(output_p1, output_p2).sum(1)
        # distances_d = (torch.mul(output_c1, output_p1).sum(1) - torch.mul(output_c2, output_p2).sum(1)).abs()
        # losses_c = F.relu(self.margin - distances_c)
        losses_p = distances_p
        losses_d = distances_d
        losses = losses_p
        if size_average:
            return losses_c.mean(), losses_p.mean(), losses_d.mean()
        else:
            return losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)
