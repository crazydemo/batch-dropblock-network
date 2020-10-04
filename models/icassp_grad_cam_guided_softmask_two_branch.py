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

def weights_init_deconv(m):
    classname = m.__class__.__name__
    nn.init.constant_(m.weight, 1.0)


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

class GumbelSigmoid(nn.Module):
    def __init__(self, max_T=1.0, decay_alpha=0.991):
        super(GumbelSigmoid, self).__init__()

        self.max_T = max_T
        self.decay_alpha = decay_alpha
        self.softmax = nn.Softmax(dim=-1)
        self.p_value = 1e-8

        self.register_buffer('cur_T', torch.tensor(max_T))

    def forward(self, x, epoch):
        if self.training:
            _cur_T = self.cur_T
        else:
            _cur_T = 0.03

        # Shape <x> : [N, 1, H, W]
        # Shape <r> : [N, 1, H, W]
        # x = x.unsqueeze(1)
        bs, c, h, w = x.size()
        x = x.view(bs, -1)
        r = 1 - x
        x = (x + self.p_value).log()
        r = (r + self.p_value).log()

        # Generate Noise
        x_N = torch.rand_like(x)
        r_N = torch.rand_like(r)
        x_N = -1 * (x_N + self.p_value).log()
        r_N = -1 * (r_N + self.p_value).log()

        # Get Final Distribution
        x = x + x_N
        x = x / (_cur_T + self.p_value)
        r = r + r_N
        r = r / (_cur_T + self.p_value)

        #x = torch.cat((x, r), dim=1)
        x = self.softmax(x) #* h * w
        # x = x[:, [1], :, :]

        if self.training:
            self.cur_T = torch.tensor((self.decay_alpha)**epoch).cuda()

        return x.view(bs, c, h, w)


class GetGrad():
    def __init__(self):
        pass

    def get_grad(self, grad):
        self.grad = grad

    def __call__(self, x):
        x.register_hook(self.get_grad)


def gradCAM(gradfeature, scores, idxs):
    getGrad = GetGrad()
    getGrad(gradfeature)
    feature = gradfeature.detach()
    bs, c, h, w = feature.size()
    output_gradCam = []
    if torch.cuda.is_available():
        score = torch.tensor(0, dtype=torch.float).cuda()
    else:
        score = torch.tensor(0, dtype=torch.float)
    for i in range(bs):
        score += scores[i, idxs[i]]
    # score = (scores.sum()).sum()
    score.backward(retain_graph=True)
    grad = getGrad.grad.detach()
    # print('grad')
    # print(grad.size())
    # print(grad[0,0,:,:])
    weight = grad.mean(2)
    weight = weight.mean(2)
    cam_cv = []
    for i in range(bs):
        grad_cam = weight[i].reshape(1, c).mm(feature[i].reshape((c, h * w)))
        grad_cam = grad_cam.reshape(h, w)
        grad_cam = F.relu(grad_cam)
        output_gradCam.append(grad_cam)

        # cam_img = grad_cam.cpu().detach().numpy()
        # cam_img = cam_img - np.min(cam_img)
        # cam_img = cam_img / np.max(cam_img)
        # cam_img = np.uint8(255 * cam_img)
        cam_img = grad_cam.detach()
        # b, c, h, w = cam_img.size()
        # cam_img = cam_img.view([b, c, -1])
        cam_img = cam_img - torch.min(cam_img)
        cam_img = cam_img / torch.max(cam_img)
        # cam_img = cam_img.view([b, c, h, w])
        cam_cv.append(cam_img)
    cam_cv = torch.stack(cam_cv, 0)
    output_gradCam = torch.stack(output_gradCam, dim=0)
    return output_gradCam, cam_cv


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
        self.res_part1 = Bottleneck(2048, 512)

        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_softmax = nn.Linear(512, num_classes)
        self.global_softmax.apply(weights_init_kaiming)
        self.global_reduction = copy.deepcopy(reduction)
        self.global_reduction.apply(weights_init_kaiming)

        # mask branch
        # self.masknet = MaskNet(2048)

        # part branch
        self.res_part2 = Bottleneck(2048, 512)

        self.part_maxpool = nn.AdaptiveMaxPool2d((1,1))
        self.batch_crop = BatchDrop(height_ratio, width_ratio)
        self.reduction = nn.Sequential(
            nn.Conv2d(2048, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        self.reduction.apply(weights_init_kaiming)
        self.softmax = nn.Linear(1024, num_classes)
        self.softmax.apply(weights_init_kaiming)


    def forward(self, x, epoch=None, y=None):
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
        # x1 = self.res_part1(x)
        glob1 = self.global_avgpool(x)
        global_triplet_feature = self.global_reduction(glob1).squeeze()
        global_softmax_class = self.global_softmax(global_triplet_feature)
        softmax_features.append(global_softmax_class)
        triplet_features.append(global_triplet_feature)
        predict.append(global_triplet_feature)

        # mask branch
        if self.training:
            p_mask, _ = gradCAM(x, global_softmax_class, y)
            p_mask = p_mask.unsqueeze(1).detach()
            n_mask = 1 - p_mask
            n_mask = n_mask.detach()

        #part branch
        x2 = self.res_part2(x)
        if self.training:
            x2 = x2*n_mask

        glob2 = self.part_maxpool(x2)
        global_triplet_feature = self.reduction(glob2).squeeze()
        global_softmax_class = self.softmax(global_triplet_feature)
        softmax_features.append(global_softmax_class)
        triplet_features.append(global_triplet_feature)
        predict.append(global_triplet_feature)

        if self.training:
            return triplet_features, softmax_features, p_mask, n_mask
        else:
            return torch.cat(predict, -1)

    def get_optim_policy(self):
        params = [
            {'params': self.backbone.parameters()},
            {'params': self.res_part.parameters()},
            # {'params': self.res_part1.parameters()},
            {'params': self.global_reduction.parameters()},
            {'params': self.global_softmax.parameters()},
            {'params': self.res_part2.parameters()},
            {'params': self.reduction.parameters()},
            {'params': self.softmax.parameters()},
            # {'params': self.masknet.parameters()},
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


class MaskNet(nn.Module):
    def __init__(self, in_planes):
        super(MaskNet, self).__init__()
        self.in_planes = in_planes
        self.backbone = nn.Sequential(
            nn.Conv2d(self.in_planes, self.in_planes//2, 1),
            nn.BatchNorm2d(self.in_planes//2),
            nn.ReLU(),
            nn.Conv2d(self.in_planes//2, 1, 1),
            nn.Sigmoid(),
        )
        self.gumble_softmax = GumbelSigmoid()
        # self.conv_x = nn.Conv2d(2048, 1, 1)
        # self.conv_y = nn.Conv2d(2048, 1, 1)
        # self.conv_x_asy = nn.Conv2d(2048, 1, (1,3), padding=(0, 1))
        # self.conv_y_asy = nn.Conv2d(2048, 1, (3,1), padding=(1, 0))
        # self.gumble_softmax = GumbelSigmoid()
        # self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU()
        # self.conv_x.apply(weights_init_kaiming)
        # self.conv_y.apply(weights_init_kaiming)
        # self.conv_x_asy.apply(weights_init_kaiming)
        # self.conv_y_asy.apply(weights_init_kaiming)
        self.backbone.apply(weights_init_kaiming)

    def forward(self, x, epoch):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        out = self.backbone(x)
        mask = out.sum(-1, keepdim=True)
        mask = mask.view([-1, 1, mask.size()[-2] // 4, 4, 1])
        mask = mask.sum(dim=-2)
        # bs, c, h, w = x.size()
        # fx_att = self.sigmoid(self.conv_x(x))
        # fy_att = self.sigmoid(self.conv_y(x))
        # fx_att = fx_att / fx_att.sum(dim=2, keepdim=True)
        # fy_att = fy_att / fy_att.sum(dim=3, keepdim=True)
        # #attention
        # fx = (x * fx_att).sum(dim=2, keepdim=True)
        # fy = (x * fy_att).sum(dim=3, keepdim=True)
        # #conv31
        # fx = self.relu(self.conv_x_asy(fx))
        # fy = self.relu(self.conv_y_asy(fy))
        # # fx = fx.view([-1, 1, 1, fx.size()[-1] // 2, 2])
        # # fx = fx.sum(dim=-1)
        # fy = fy.view([-1, 1, fy.size()[-2] // 4, 4, 1])
        # fy = fy.sum(dim=-2)
        # fx = fx.repeat([1, 1, 2, 1])
        # fx = fx.permute(0, 1, 3, 2)#.view(bs, 1, 1, -1)
        # fx = fx.reshape([bs, 1, 1, -1])
        # fy = fy.repeat([1, 1, 1, 4])
        # fy = fy.reshape([bs, 1, -1, 1])
        # mask = fy
        # mask = self.sigmoid(mask)
        mask = self.gumble_softmax(mask, epoch)
        bs, c, h, w = mask.size()
        res = mask.unsqueeze(-2)
        res = res.repeat(1, 1, 1, 4, 1)
        res = res.reshape([bs, 1, h*4, w])

        # res = res.unsqueeze(-2)
        res = res.repeat(1, 1, 1, 8)
        # res = res.permute(0, 1, 2, 4, 3)
        # res = res.reshape([bs, 1, h*4, 2*w])

        return res

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
