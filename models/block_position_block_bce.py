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
import cv2
import os

from .resnet import ResNet

img_path = '/media/ivy/research/BDB_raw/batch-dropblock-network/experiment_res/spatial_based_baseline_gap_first/block_position_spatial_logits_for_draw/market/imgs/epoch200/'

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
            nn.Conv2d(2048, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        # block id
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_softmax = nn.Conv2d(1024, num_classes, 1)
        self.global_softmax.apply(weights_init_kaiming)
        self.global_reduction = copy.deepcopy(reduction)
        self.global_reduction.apply(weights_init_kaiming)

        #block position
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        self.blk_position = nn.Sequential(
            nn.Conv2d(2048, 1024, 5, 2, 2),
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Conv2d(1024, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 1, 1),
        )
        self.blk_position.apply(weights_init_kaiming)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)

        '''
        # part branch
        self.res_part2 = Bottleneck(2048, 512)

        self.part_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.batch_crop = BatchDrop(height_ratio, width_ratio)
        self.reduction = nn.Sequential(
            nn.Linear(2048, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.reduction.apply(weights_init_kaiming)
        self.softmax = nn.Linear(1024, num_classes)
        self.softmax.apply(weights_init_kaiming)
        '''

    def forward(self, x, y=None):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        img = x
        x = self.backbone(x)
        x = self.res_part(x)

        predict = []
        triplet_features = []
        softmax_features = []

        #block positions
        x_upsampled = self.upsample(x)
        predicted_mask = self.blk_position(x_upsampled)
        bs, c, h, w = predicted_mask.size()
        predicted_mask = predicted_mask.squeeze()
        step = [3, 2]
        cam_patchs_row = torch.split(predicted_mask, step[0], 1)
        patchs_row = torch.stack(cam_patchs_row, 1)
        cam_patchs_col = torch.split(patchs_row, step[1], 3)
        patchs_col = torch.cat(cam_patchs_col, 1)
        patchs = patchs_col.sum(dim=[2, 3])  # N*12
        predicted_mask_for_test = self.softmax(patchs).detach()
        predicted_mask_ = self.sigmoid(patchs)

        #block id label mask
        x_detached = x.detach()
        label_mask = self.global_reduction(x_detached)
        label_mask = self.global_softmax(label_mask)
        label_mask = label_mask.detach()
        if self.training:
            label_lst = []
            for i in range(x.size()[0]):
                score = label_mask[i, y[i], :, :]
                label_lst.append(score)
            label_mask = torch.stack(label_lst, 0)

            _, f_h, f_w = label_mask.size()
            step = [3, 2]
            cam_patchs_row = torch.split(label_mask, step[0], 1)
            patchs_row = torch.stack(cam_patchs_row, 1)
            cam_patchs_col = torch.split(patchs_row, step[1], 3)
            patchs_col = torch.cat(cam_patchs_col, 1)
            patchs = patchs_col.sum(dim=[2, 3])  # N*12
            element, _ = patchs.sort(descending=True)
            threshold_idx = int(element.size()[-1] * 0.33)
            threshold = element[threshold_idx]
            label_mask = element.ge(threshold)
                # if not os.path.exists(img_path):
                #     os.makedirs(img_path)
                # predicted_mask = predicted_mask_.view(bs, 1, h, w)
                # label_saved = score.view(h, w)
                # img_raw = img[i, :, :, :].detach().cpu().numpy()
                # img_raw = img_raw.transpose(1,2,0)
                # img_raw = img_raw - np.min(img_raw)
                # img_raw = img_raw / np.max(img_raw)
                # img_raw = np.uint8(255 * img_raw)
                # img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
                # label_saved = label_saved.cpu().numpy()
                # label_saved = label_saved - np.min(label_saved)
                # label_saved = label_saved / np.max(label_saved)
                # label_saved = np.uint8(255 * label_saved)
                # predicted_saved = predicted_mask[i,0,:,:].detach().cpu().numpy()
                # predicted_saved = predicted_saved - np.min(predicted_saved)
                # predicted_saved = predicted_saved / np.max(predicted_saved)
                # predicted_saved = np.uint8(255 * predicted_saved)
                # label_saved = np.expand_dims(label_saved, -1)
                # predicted_saved = np.expand_dims(predicted_saved, -1)
                # label_saved = cv2.resize(label_saved, (128,384))
                # predicted_saved = cv2.resize(predicted_saved, (128, 384))
                # cv2.imwrite(img_path + 'img' + str(i) + '.jpg', img_raw)
                # cv2.imwrite(img_path + 'label'+str(i)+'.jpg', label_saved)
                # cv2.imwrite(img_path + 'predicted'+str(i)+'.jpg', predicted_saved)
                # print('saving done!')



            predicted_mask_ = predicted_mask_.view(x.size()[0], -1)
            label_mask_ =label_mask.view(x.size()[0], -1)

        predicted_mask_for_test = predicted_mask_for_test.unsqueeze(1)
        step = [3, 2]
        cam_patchs_row = torch.split(x, step[0], 2)
        patchs_row = torch.stack(cam_patchs_row, 2)
        cam_patchs_col = torch.split(patchs_row, step[1], 4)
        patchs_col = torch.cat(cam_patchs_col, 2)
        patchs = patchs_col.sum(dim=[3, 4])  # N*12
        x = torch.sum(patchs * predicted_mask_for_test, dim=-1, keepdim=True)
        x = x.unsqueeze(-1)
        global_triplet_feature = self.global_reduction(x)
        global_softmax_class = self.global_softmax(global_triplet_feature)
        softmax_features.append(global_softmax_class.squeeze())
        triplet_features.append(global_triplet_feature.squeeze())
        predict.append(global_triplet_feature.squeeze())

        if self.training:
            return triplet_features, softmax_features, predicted_mask_, label_mask_.float()
        else:
            return global_triplet_feature.squeeze()

    def get_optim_policy(self):
        params = [
            {'params': self.backbone.parameters()},
            {'params': self.res_part.parameters()},
            {'params': self.global_reduction.parameters()},
            {'params': self.global_softmax.parameters()},
            # {'params': self.res_part2.parameters()},
            # {'params': self.reduction.parameters()},
            # {'params': self.softmax.parameters()},
            {'params': self.blk_position.parameters()},
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