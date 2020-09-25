# encoding: utf-8
import random
import torch
from torch import nn
import torch.nn.functional as F
import math

def topk_mask(input, dim, K = 10, **kwargs):
    index = input.topk(max(1, min(K, input.size(dim))), dim = dim, **kwargs)[1]
    return torch.autograd.Variable(torch.zeros_like(input.data)).scatter(dim, index, 1.0)

def pdist(A, squared = False, eps = 1e-4):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min = 0)
    return res if squared else res.clamp(min = eps).sqrt()


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def dist_after_cross_attention(x, p_idx, n_idx):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    dist_ap = []
    dist_an = []
    for i in range(len(p_idx)):
        a = normalize(x[i, :, :, :], axis=0)
        p = normalize(x[p_idx[i], :, :, :], axis=0)
        n = normalize(x[n_idx[i], :, :, :], axis=0)
        # a, p, n = x[i, :, :, :], x[p_idx[i], :, :, :], x[n_idx[i], :, :, :]
        a_ = cross_attention(a, a)
        p_ = cross_attention(a, p)
        dist_ap.append(torch.sum((a_ - p_) ** 2).clamp(min=1e-12).sqrt())
        n_ = cross_attention(a, n)
        dist_an.append(torch.sum((a_ - n_) ** 2).clamp(min=1e-12).sqrt())
    return dist_ap, dist_an

def cross_attention(anchor, output):
    c, h, w = anchor.size()
    anchor = anchor.view(c, h*w)
    output = output.view(c, h*w)
    anchor = anchor.t()
    attention = anchor.mm(output)
    attention = F.softmax(attention, 0)
    output_ = output.mm(attention)
    output1 = torch.mean(output_, -1)
    output2 = torch.mean(output, -1)
    diff = torch.sum(output2-output1)
    output = output1+output2
    '''
    anchor = anchor.t()
    anchor_ = anchor.mm(attention)
    anchor1 = torch.mean(anchor_, -1)
    anchor2 = torch.mean(anchor, -1)
    diff = torch.sum(anchor2 - anchor1)
    anchor = anchor1 + anchor2
    return anchor, output
    '''
    return output
def hard_example_mining(dist_mat, labels, margin, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    torch.set_printoptions(threshold=5000) 
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels, self.margin)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an

class TripletLoss_pp(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""


    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat1 = normalize(global_feat[:, :512], axis=-1)
            global_feat2 = normalize(global_feat[:, 512:], axis=-1)
        else:
            global_feat1 = global_feat[:, :512]
            global_feat2 = global_feat[:, 512:]
        dist_mat1 = euclidean_dist(global_feat1, global_feat1)
        dist_mat2 = euclidean_dist(global_feat2, global_feat2)
        dist_mat = dist_mat1 + dist_mat2
        dist_ap, dist_an = hard_example_mining(dist_mat, labels, self.margin)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an

class TripletLoss_cross_attention(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, score_tensor, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        _, dist_an, p_inds, n_inds = hard_example_mining(dist_mat, labels, self.margin, True)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        dist_ap, _ = dist_after_cross_attention(score_tensor, p_inds, n_inds)
        dist_ap = torch.tensor(dist_ap).cuda()
        dist_an = torch.tensor(dist_an).cuda()
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class LikelihoodLoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, a=0.001, lamda=0.001, use_gpu=True):#a=0.001 for lgm
        super(LikelihoodLoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.use_gpu = use_gpu
        self.lamda = lamda
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, score, label, ):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        ALPHA = torch.zeros(score.size()).scatter_(1, label.unsqueeze(1).data.cpu(), 1)
        max_num = torch.max(ALPHA)
        K = ALPHA * self.a + 1
        logits_with_margin = score * K.cuda()
        likelihood = self.lamda * torch.mean(torch.sum(-1. * score * ALPHA.cuda(), 1))

        return likelihood, logits_with_margin


class Margin:
    def __call__(self, embeddings, labels):
        embeddings = F.normalize(embeddings)
        alpha = 0.2
        beta = 1.2
        distance_threshold = 0.5
        inf = 1e6
        eps = 1e-6
        distance_weighted_sampling = True
        d = pdist(embeddings)
        pos = torch.eq(*[labels.unsqueeze(dim).expand_as(d) for dim in [0, 1]]).type_as(d) - torch.autograd.Variable(torch.eye(len(d))).type_as(d)
        num_neg = int(pos.data.sum() / len(pos))
        if distance_weighted_sampling:
            '''
            dim = embeddings.size(-1)
            distance = d.data.clamp(min = distance_threshold)
            distribution = distance.pow(dim - 2) * ((1 - distance.pow(2) / 4).pow(0.5 * (dim - 3)))
            weights = distribution.reciprocal().masked_fill_(pos.data + torch.eye(len(d)).type_as(d.data) > 0, eps)
            samples = torch.multinomial(weights, replacement = False, num_samples = num_neg)
            neg = torch.autograd.Variable(torch.zeros_like(pos.data).scatter_(1, samples, 1))
            '''
            neg = torch.autograd.Variable(torch.zeros_like(pos.data).scatter_(1, torch.multinomial((d.data.clamp(min = distance_threshold).pow(embeddings.size(-1) - 2) * (1 - d.data.clamp(min = distance_threshold).pow(2) / 4).pow(0.5 * (embeddings.size(-1) - 3))).reciprocal().masked_fill_(pos.data + torch.eye(len(d)).type_as(d.data) > 0, eps), replacement = False, num_samples = num_neg), 1))
        else:
            neg = topk_mask(d  + inf * ((pos > 0) + (d < distance_threshold)).type_as(d), dim = 1, largest = False, K = num_neg)
        L = F.relu(alpha + (pos * 2 - 1) * (d - beta))
        M = ((pos + neg > 0) * (L > 0)).float()
        return (M * L).sum() / M.sum(), 0

