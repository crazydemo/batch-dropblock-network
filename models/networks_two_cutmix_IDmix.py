import copy
import itertools

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import random
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from torch import nn, optim
from torch.utils.data import dataloader
from torchvision import transforms
from torchvision.models.resnet import Bottleneck, resnet50
from torchvision.transforms import functional

from .resnet import ResNet
import cv2


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


def number_of_certain_probability(sequence, probability):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(sequence, probability):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


class CutMixBatchDrop(nn.Module):
    def __init__(self, h_ratio, w_ratio):
        super(CutMixBatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x, cam, y, step, k, randy=None):
        '''
        :param x: feature map with shape[N,2048,24,8]
        :param cam: grad cam map with shape[N,24,8]
        :param y: gt of x, [N,1]
        :return: x_new = mask*x1 + (1-mask)*x2
        '''
        if self.training:
            bs, c, h, w = x.size()
            mask = torch.ones(x.size()).cuda()

            '''get mask'''
            _, f_h, f_w = cam.size()
            cam_patchs_row = torch.split(cam, step[0], 1)
            patchs_row = torch.stack(cam_patchs_row, 1)
            cam_patchs_col = torch.split(patchs_row, step[1], 3)
            patchs_col = torch.cat(cam_patchs_col, 1)
            patchs = patchs_col.sum(dim=[2, 3])  # N*12
            patchs, patchs_idxs = patchs.sort(1, True)
            # l = int(f_h / step[0])
            for i in range(bs):
                for idx in patchs_idxs[i, 0:k]:
                    if idx < 8:
                        mask[i, :, idx * step[0]:idx * step[0] + step[0], 0:2] = 0
                    elif idx < 16:
                        mask[i, :, (idx - 8) * step[0]:(idx - 8) * step[0] + step[0], 2:4] = 0
                    elif idx < 24:
                        mask[i, :, (idx - 16) * step[0]:(idx - 16) * step[0] + step[0], 4:6] = 0
                    else:
                        mask[i, :, (idx - 24) * step[0]:(idx - 24) * step[0] + step[0], 6:8] = 0

                # print(mask[i,0,:,:])

            '''CutMix'''
            lamda = k * step[0] * step[1] / (h * w)
            bs, c, h, w = x.size()
            if randy is not None:
                rand_idx = randy
                y2 = y[randy]
                # y_temp = y.reshape([-1, 4])
                # rand_idx = randy
                # y2 = y_temp[:, rand_idx].reshape([-1, 1])
                # x2 = x.reshape([-1, 4, c, h, w])
                # x2 = x2[:,rand_idx,:,:,:].reshape([bs,c,h,w])
            else:
                # 随机mix
                # rand_idx = torch.randperm(bs)

                # 同类mix
                '''
                rand_idx = []
                for i in range(bs // 4):
                    temp = torch.randperm(4) + i * 4
                    # print(temp)
                    rand_idx += temp.numpy().tolist()
                '''

                # 不同类mix

                idx = list(range(bs))
                rand_idx = []
                for i in range(bs // 4):
                    idx_part = idx[i * 4:(i + 1) * 4]
                    temp = list(set(idx) - set(idx_part))
                    rand_idx += [random.choice(temp) for k in range(4)]

                # y2 = y[rand_idx]

            # y_new = lamda*y + (1-lamda)*y2
            x2 = x[rand_idx, :, :, :]
            x_new = (mask * x + (torch.ones(mask.size()).cuda() - mask) * x2)

        return x_new, rand_idx, lamda


def getCAM(feature, weights, idxs):
    bs, c, h, w = feature.size()
    output_cam = []
    cam_cv =[]
    for i in range(bs):
        cam = weights[idxs[i]].reshape(1, c).mm(feature[i].reshape((c, h * w)))
        cam = cam.reshape(h, w)
        output_cam.append(cam)

        cam_img = cam.cpu().detach().numpy()
        cam_img = cam_img - np.min(cam_img)
        cam_img = cam_img / np.max(cam_img)
        cam_img = np.uint8(255 * cam_img)
        cam_cv.append(cam_img)

    cam_cv = np.array(cam_cv)
    output_cam = torch.stack(output_cam, dim=0)
    return output_cam, cam_cv


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

        cam_img = grad_cam.cpu().detach().numpy()
        cam_img = cam_img - np.min(cam_img)
        cam_img = cam_img / np.max(cam_img)
        cam_img = np.uint8(255 * cam_img)
        cam_cv.append(cam_img)

    '''
    for i in range(bs):
        score = scores[i, idxs[i]]
        print('score')
        print(score)
        score.backward()
        grad = getGrad.grad.detach()
        print('grad')
        print(grad.size())
        print(grad[0,0,:,:])
        print(grad[2,0,:,:])
        print(grad[2,2,:,:])
        weight = grad[i].mean(1)
        weight = weight.mean(1)
        # print('weight')
        # print(weight.size())
        grad_cam = weight.reshape(1, c).mm(feature[i].reshape((c, h * w)))
        grad_cam = grad_cam.reshape(h, w)
        grad_cam = F.relu(grad_cam)
        output_gradCam.append(grad_cam)
    '''
    cam_cv = np.array(cam_cv)
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
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )
        # Resnet50 stage4
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
            nn.Conv2d(2048, 512, 1),  # def __init__(self, in_channels, out_channels, kernel_size, stride=1,
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # global branch
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_softmax = nn.Linear(512, num_classes)
        self.global_softmax.apply(weights_init_kaiming)
        self.global_reduction = copy.deepcopy(reduction)
        self.global_reduction.apply(weights_init_kaiming)

        # part branch
        # self.bottleneck1 = Bottleneck(2048, 512)
        # self.bottleneck2 = Bottleneck(2048, 512)
        self.res_part2 = Bottleneck(2048, 512)
        self.part_maxpool = nn.AdaptiveMaxPool2d((1, 1))

        # cutmix branch1
        self.cutmix_batch_drop1 = CutMixBatchDrop(height_ratio, width_ratio)
        self.reduction1 = nn.Sequential(
            nn.Linear(2048, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.reduction1.apply(weights_init_kaiming)
        self.softmax1 = nn.Linear(1024, num_classes)
        self.softmax1.apply(weights_init_kaiming)

        # cutmix branch2
        # self.bottleneck2 = Bottleneck(2048, 512)
        self.cutmix_batch_drop2 = CutMixBatchDrop(height_ratio, width_ratio)
        self.reduction2 = nn.Sequential(
            nn.Linear(2048, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.reduction2.apply(weights_init_kaiming)
        self.softmax2 = nn.Linear(512, num_classes)
        self.softmax2.apply(weights_init_kaiming)

    def forward(self, x, y=None):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        x_img = x.cpu().detach().numpy()
        N, _, camH, camW = x.size()

        x = self.backbone(x)  # N*1024*24*8
        x = self.res_part(x)  # N*2048*24*8
        cam_global_feature = x.detach()
        grad_cam = x

        predict = []
        triplet_features = []
        softmax_features = []

        # global branch
        glob = self.global_avgpool(x)  # GAP--N*2048*1
        global_triplet_feature = self.global_reduction(glob).squeeze()  # N*512
        global_softmax_class = self.global_softmax(global_triplet_feature)  # N*num_class
        softmax_features.append(global_softmax_class)
        triplet_features.append(global_triplet_feature)
        predict.append(global_triplet_feature)

        # part branch
        x = self.res_part2(x)  # N*2048*24*8
        cam_cutmix_feature = x.detach()
        # x_cutmix1 = self.bottleneck1(x)
        # x_cutmix2 = self.bottleneck2(x)
        x_cutmix1 = x
        x_cutmix2 = x
        '''get grad-cam map'''
        if self.training:
            # cutmix branch1
            gradcam, cam_cv = gradCAM(grad_cam, global_softmax_class, y)
            x_cutmix1, idx_cutmix1, lamda = self.cutmix_batch_drop1(x_cutmix1, gradcam, y, [3, 2], 6)
            cutmix1_triplet_feature = self.part_maxpool(x_cutmix1).squeeze()  # N*2048
            cutmix1_feature = self.reduction1(cutmix1_triplet_feature)  # N*1024
            cutmix1_softmax_feature = self.softmax1(cutmix1_feature)  # N*num_class/751

            feature_img = x_cutmix1.cpu().detach().numpy()
            for i in range(N):
                heatmap = cv2.applyColorMap(cv2.resize(cam_cv[i], (camW, camH)), cv2.COLORMAP_JET)
                img = feature_img[i, 0, :, :] - np.min(feature_img[i, 0, :, :])
                # img = x_img[i] - np.min(x_img[i])
                img = img / np.max(img)
                # img = np.uint8(255 * img).transpose(1, 2, 0)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (camW, camH))
                # result = heatmap * 0.5 + img * 0.5
                cv2.imwrite('cam-img/feature-img/two_cutmix1/Heat{}.jpg'.format(i), heatmap)
                cv2.imwrite('cam-img/feature-img/two_cutmix1/feature{}.jpg'.format(i), img)
                # cv2.imwrite('feature-img/two_cutmix_best/cutmix1/Heatimg{}.jpg'.format(i), result)
            print('ok')

            # cutmix branch2
            grad_feature = x_cutmix1
            gradcam_cutmix, cam_cv = gradCAM(grad_feature, cutmix1_softmax_feature, y)
            x_cutmix2, idx_cutmix2, _ = self.cutmix_batch_drop2(x_cutmix2, gradcam_cutmix, y, [3, 2], 6)

            # y_temp = y.reshape([-1, 4])
            # rand_idx = idx_cutmix2
            # y2 = y_temp[:, rand_idx].reshape([-1, 1])

            feature_img = x_cutmix2.cpu().detach().numpy()
            for i in range(N):
                heatmap = cv2.applyColorMap(cv2.resize(cam_cv[i], (camW, camH)), cv2.COLORMAP_JET)
                # img = x_img[i] - np.min(x_img[i])
                img = feature_img[i, 0, :, :] - np.min(feature_img[i, 0, :, :])
                img = img / np.max(img)
                # img = np.uint8(255 * img).transpose(1, 2, 0)
                img = cv2.resize(img, (camW, camH))
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # result = heatmap * 0.5 + img * 0.5
                cv2.imwrite('cam-img/feature-img/two_cutmix2/Heat{}.jpg'.format(i), heatmap)
                cv2.imwrite('cam-img/feature-img/two_cutmix2/feature{}.jpg'.format(i), img)
                # cv2.imwrite('feature-img/two_cutmix_best/cutmix2/Heatimg{}.jpg'.format(i), result)
            print('ok')

            '''
            feature_img = x_cutmix2.cpu().detach().numpy()
            for i in range(N):
                heatmap = cv2.applyColorMap(cv2.resize(cam_cv[i], (camW, camH)), cv2.COLORMAP_JET)
                # img = feature_img[i,0,:,:] - np.min(feature_img[i,0,:,:])
                img = x_img[i] - np.min(x_img[i])
                img = img / np.max(img)
                img = np.uint8(255 * img).transpose(1, 2, 0)
                # img = cv2.resize(img, (camW,camH))
                result = heatmap * 0.5 + img * 0.5
                cv2.imwrite('cam-img/two_cutmix/cutmix2/CAM{}.jpg'.format(i), heatmap)
                cv2.imwrite('cam-img/two_cutmix/cutmix2/img{}.jpg'.format(i), img)
                cv2.imwrite('cam-img/two_cutmix/cutmix2/cam-img{}.jpg'.format(i), result)
            print('ok')
            '''

        '''show img'''
        # global
        paras_re = list(self.global_reduction.parameters())
        para_re1 = paras_re[0].detach().squeeze()
        para_re2 = paras_re[2].detach().squeeze()
        paras_soft = list(self.global_softmax.parameters())
        para_soft = paras_soft[0].detach().squeeze()
        weights_cam = para_soft.mm(para_re2.reshape(512, 1) * para_re1)  # 751*2048
        cam_global, cam_cv = getCAM(cam_global_feature, weights_cam, y)  # N*24*8
        # N, H, W = cam_global.size()
        for i in range(N):
            heatmap = cv2.applyColorMap(cv2.resize(cam_cv[i], (camW, camH)), cv2.COLORMAP_JET)
            # img = feature_img[i,0,:,:] - np.min(feature_img[i,0,:,:])
            img = x_img[i] - np.min(x_img[i])
            img = img / np.max(img)
            img = np.uint8(255 * img).transpose(1, 2, 0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img, (camW,camH))
            result = heatmap * 0.5 + img * 0.5
            cv2.imwrite('cam-img/two_cutmix_global/CAM{}.jpg'.format(i), heatmap)
            cv2.imwrite('cam-img/two_cutmix_global/img{}.jpg'.format(i), img)
            cv2.imwrite('cam-img/two_cutmix_global/cam-img{}.jpg'.format(i), result)
        print('ok')

        # cutmix1
        paras_re = list(self.reduction1.parameters())
        para_re1 = paras_re[0].detach().squeeze()
        para_re2 = paras_re[2].detach().squeeze()
        paras_soft = list(self.softmax1.parameters())
        para_soft = paras_soft[0].detach().squeeze()
        weights_cam = para_soft.mm(para_re2.reshape(1024, 1) * para_re1)  # 751*2048
        cam_cutmix1, cam_cv = getCAM(cam_cutmix_feature, weights_cam, y)  # N*24*8
        for i in range(N):
            heatmap = cv2.applyColorMap(cv2.resize(cam_cv[i], (camW, camH)), cv2.COLORMAP_JET)
            # img = feature_img[i,0,:,:] - np.min(feature_img[i,0,:,:])
            img = x_img[i] - np.min(x_img[i])
            img = img / np.max(img)
            img = np.uint8(255 * img).transpose(1, 2, 0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = heatmap * 0.5 + img * 0.5
            cv2.imwrite('cam-img/two_cutmix1/CAM{}.jpg'.format(i), heatmap)
            cv2.imwrite('cam-img/two_cutmix1/img{}.jpg'.format(i), img)
            cv2.imwrite('cam-img/two_cutmix1/cam-img{}.jpg'.format(i), result)
        print('ok')

        # cutmix2
        paras_re = list(self.reduction2.parameters())
        para_re1 = paras_re[0].detach().squeeze()
        para_re2 = paras_re[2].detach().squeeze()
        paras_soft = list(self.softmax2.parameters())
        para_soft = paras_soft[0].detach().squeeze()
        weights_cam = para_soft.mm(para_re2.reshape(512, 1) * para_re1)  # 751*2048
        cam_cutmix2, cam_cv = getCAM(cam_cutmix_feature, weights_cam, y)  # N*24*8
        for i in range(N):
            heatmap = cv2.applyColorMap(cv2.resize(cam_cv[i], (camW, camH)), cv2.COLORMAP_JET)
            img = x_img[i] - np.min(x_img[i])
            img = img / np.max(img)
            img = np.uint8(255 * img).transpose(1, 2, 0)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = heatmap * 0.5 + img * 0.5
            cv2.imwrite('cam-img/two_cutmix2/CAM{}.jpg'.format(i), heatmap)
            cv2.imwrite('cam-img/two_cutmix2/img{}.jpg'.format(i), img)
            cv2.imwrite('cam-img/two_cutmix2/cam-img{}.jpg'.format(i), result)
        print('ok')

        # cutmix1 branch
        cutmix1_triplet_feature = self.part_maxpool(x_cutmix1).squeeze()  # N*2048
        cutmix1_feature = self.reduction1(cutmix1_triplet_feature)  # N*1024
        cutmix1_softmax_feature = self.softmax1(cutmix1_feature)  # N*num_class/751
        triplet_features.append(cutmix1_feature)
        softmax_features.append(cutmix1_softmax_feature)
        predict.append(cutmix1_feature)

        # cutmix2 branch
        cutmix2_triplet_feature = self.part_maxpool(x_cutmix2).squeeze()  # N*2048
        cutmix2_feature = self.reduction2(cutmix2_triplet_feature)  # N*512
        cutmix2_softmax_feature = self.softmax2(cutmix2_feature)  # N*num_class/751
        triplet_features.append(cutmix2_feature)
        softmax_features.append(cutmix2_softmax_feature)
        predict.append(cutmix2_feature)

        if self.training:
            return triplet_features, softmax_features, y[idx_cutmix1]
        else:
            return torch.cat(predict, 1)

    def get_optim_policy(self):
        params = [
            {'params': self.backbone.parameters()},
            {'params': self.res_part.parameters()},
            {'params': self.global_reduction.parameters()},
            {'params': self.global_softmax.parameters()},
            # {'params': self.bottleneck1.parameters()},
            # {'params': self.bottleneck2.parameters()},
            {'params': self.res_part2.parameters()},
            {'params': self.reduction1.parameters()},
            {'params': self.softmax1.parameters()},
            {'params': self.reduction2.parameters()},
            {'params': self.softmax2.parameters()},
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





