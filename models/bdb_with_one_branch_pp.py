# encoding: utf-8
import copy
import itertools

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import random
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from torch import nn, optim
from torch.utils.data import dataloader
from torchvision import transforms
from torchvision.models.resnet import Bottleneck, resnet50
from torchvision.transforms import functional

from .resnet import ResNet

def LGM(z_mu, c_mu, c_var):
    epsi = 1e-6

    z_mu = torch.unsqueeze(z_mu, 1)
    c_mu = torch.unsqueeze(c_mu, 0)
    c_var = torch.unsqueeze(c_var, 0)
    diff = (z_mu-c_mu)**2
    neg_dist = torch.sum(diff / (c_var+epsi), -1)#batchsize, num_class
    return neg_dist


def KL_between_multivariate_gaussian(z_mu, z_var, c_mu, c_var):
    epsi = 1e-6

    det_z_var = torch.prod(z_var,1)
    det_c_var = torch.prod(c_var,1)
    inverse_c_var = 1 / (c_var+epsi)

    det_term = torch.unsqueeze(torch.log(det_c_var+epsi), 0) - torch.unsqueeze(torch.log(det_z_var+epsi), 1) #batchsize, num_class
    trace_term = torch.mm(z_var, inverse_c_var.t())#batchsize, num_class
    z_mu = torch.unsqueeze(z_mu, 1)
    c_mu = torch.unsqueeze(c_mu, 0)
    c_var = torch.unsqueeze(c_var, 0)
    diff = (z_mu-c_mu)**2
    m_dist_term = torch.sum(diff / (c_var+epsi), -1)#batchsize, num_class

    KL_divergence = 0.5*(det_term+trace_term+m_dist_term)
    return KL_divergence

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BatchDrop(nn.Module):
    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h - rh)
            sy = random.randint(0, w - rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx + rh, sy:sy + rw] = 0
            x = x * mask
        return x


class BatchCrop(nn.Module):
    def __init__(self, ratio):
        super(BatchCrop, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rw = int(self.ratio * w)
            start = random.randint(0, h - 1)
            if start + rw > h:
                select = list(range(0, start + rw - h)) + list(range(start, h))
            else:
                select = list(range(start, start + rw))
            mask = x.new_zeros(x.size())
            mask[:, :, select, :] = 1
            x = x * mask
        return x


class ResNetBuilder(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes=None, last_stride=1, pretrained=False):
        super().__init__()
        self.base = ResNet(last_stride)
        if pretrained:
            model_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
            self.base.load_param(model_zoo.load_url(model_url))

        self.num_classes = num_classes
        if num_classes is not None:
            self.bottleneck = nn.Sequential(
                nn.Linear(self.in_planes, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.1),
                nn.Dropout(p=0.5)
            )
            self.bottleneck.apply(weights_init_kaiming)
            self.classifier = nn.Linear(512, self.num_classes)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        global_feat = self.base(x)
        global_feat = F.avg_pool2d(global_feat, global_feat.shape[2:])  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)
        if self.training and self.num_classes is not None:
            feat = self.bottleneck(global_feat)
            cls_score = self.classifier(feat)
            return [global_feat], [cls_score]
        else:
            return global_feat

    def get_optim_policy(self):
        base_param_group = self.base.parameters()
        if self.num_classes is not None:
            add_param_group = itertools.chain(self.bottleneck.parameters(), self.classifier.parameters())
            return [
                {'params': base_param_group},
                {'params': add_param_group}
            ]
        else:
            return [
                {'params': base_param_group}
            ]


class SigmaNet(nn.Module):
    def __init__(self, indim, outdim):
        super(SigmaNet, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.reduction1 = nn.Sequential(
            nn.Conv2d(self.indim, self.indim//4, 1, bias=False),
            nn.BatchNorm2d(self.indim//4),
            nn.Tanh()
        )
        self.reduction2 = nn.Sequential(
            nn.Conv2d(self.indim, self.indim//4, 1, bias=False),
            nn.BatchNorm2d(self.indim//4),
            nn.Tanh()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.fusion_block = nn.Sequential(
            nn.Dropout2d(0.5),
            nn.Conv2d(self.indim, self.outdim, 1),
            nn.BatchNorm2d(self.outdim),
            nn.ReLU()
        )
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.outdim, self.outdim, bias=True)
        self.identity = nn.Conv2d(self.indim, self.outdim, 1, bias=True)
        self.softplus = nn.Softplus()

        self.reduction1.apply(weights_init_kaiming)
        self.reduction2.apply(weights_init_kaiming)
        self.fusion_block.apply(weights_init_kaiming)
        self.fc.apply(weights_init_kaiming)
        self.identity.apply(weights_init_kaiming)

    def forward(self, x):
        # b, c, h, w = x.size()
        f1 = self.reduction1(x)
        f2 = self.reduction2(x)

        f_max1 = self.maxpool(f1)
        f_min1 = -1 * self.maxpool(-1 * f1)
        f_max2 = self.maxpool(f2)
        f_min2 = -1 * self.maxpool(-1 * f2)
        f = []
        f.append(f_max1 * f_max2)
        f.append(f_max1 * f_min2)
        f.append(f_min1 * f_min2)
        f.append(f_min1 * f_max2)
        f = torch.cat(f, 1)

        f_fused = self.fusion_block(f)
        f_fused = self.global_avgpool(f_fused).squeeze()
        f_identity = self.identity(x)
        f_identity = self.global_avgpool(f_identity).squeeze()
        log_sigma = self.fc(f_fused) + f_identity
        sigma = self.softplus(log_sigma)
        return sigma

class BFE(nn.Module):
    def __init__(self, num_classes, width_ratio=0.5, height_ratio=0.5):
        super(BFE, self).__init__()
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3,  # res_conv4
        )
        self.res_part = nn.Sequential(
            Bottleneck(1024, 512, stride=1, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(2048),
            )),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512),
        )
        self.res_part.load_state_dict(resnet.layer4.state_dict())
        reduction = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # global branch
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.prior_mu = nn.Parameter(torch.randn(num_classes, 512).cuda())
        nn.init.kaiming_normal_(self.prior_mu, a=0, mode='fan_in')
        self.prior_log_sigma = nn.Parameter(torch.ones(num_classes, 512).cuda())
        self.softplus = nn.Softplus()

        # self.sigmanet = nn.Sequential(
        #     nn.Conv2d(2048, 512, 1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.Conv2d(512, 512, 1),
        #     nn.Softplus()
        # )

        self.sigmanet = SigmaNet(2048, 512)

        self.global_reduction = copy.deepcopy(reduction)
        self.global_reduction.apply(weights_init_kaiming)
        self.sigmanet.apply(weights_init_kaiming)

    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)
        x = self.res_part(x)

        predict = []
        triplet_features = []
        softmax_features = []

        # global branch
        glob = self.global_avgpool(x)
        global_triplet_feature = self.global_reduction(glob).squeeze()
        posterior_mu = global_triplet_feature
        posterior_sigma = self.sigmanet(glob).squeeze()

        prior_mu = self.prior_mu
        prior_sigma = self.softplus(self.prior_log_sigma*0.54)

        global_softmax_class = -1 * KL_between_multivariate_gaussian(posterior_mu, posterior_sigma, prior_mu, prior_sigma)

        # softmax_features.append(global_softmax_class)
        # triplet_features.append(global_triplet_feature)
        # predict.append(global_triplet_feature)


        if self.training:
            return global_softmax_class, posterior_mu, posterior_sigma, prior_mu, prior_sigma
        else:
            return torch.cat([posterior_mu, posterior_sigma],1)

    def get_optim_policy(self):
        params = [
            {'params': self.backbone.parameters()},
            {'params': self.res_part.parameters()},
            {'params': self.global_reduction.parameters()},
            {'params': self.sigmanet.parameters()},
            {'params': self.prior_mu},
            {'params': self.prior_log_sigma, 'weight_decay':0.},
        ]
        return params


class Resnet(nn.Module):
    def __init__(self, num_classes, resnet=None):
        super(Resnet, self).__init__()
        if not resnet:
            resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3,  # res_conv4
            resnet.layer4
        )
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Linear(2048, num_classes)

    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)

        x = self.global_avgpool(x).squeeze()
        feature = self.softmax(x)
        if self.training:
            return [], [feature]
        else:
            return feature

    def get_optim_policy(self):
        return self.parameters()


class IDE(nn.Module):
    def __init__(self, num_classes, resnet=None):
        super(IDE, self).__init__()
        if not resnet:
            resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3,  # res_conv4
            resnet.layer4
        )
        self.global_avgpool = nn.AvgPool2d(kernel_size=(12, 4))

    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x = self.backbone(x)

        feature = self.global_avgpool(x).squeeze()
        if self.training:
            return [feature], []
        else:
            return feature

    def get_optim_policy(self):
        return self.parameters()