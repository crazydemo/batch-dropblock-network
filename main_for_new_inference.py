# encoding: utf-8
import os
import sys
from os import path as osp
from pprint import pprint

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from config import opt
from datasets import data_manager
from datasets.data_loader import ImageData
from datasets.samplers import RandomIdentitySampler
from models.two_cutmix_shared_new_inference import ResNetBuilder, IDE, Resnet, BFE

# G_branch, G_branch_triplet_KL_xent


# bfe, bfe_channel_drop, two_cutmix, bfe_part_channel_drop, two_cutmix_with_channel_drop, two_cutmix_with_attentive_channel_cutmix
# two_cutmix_shared, block_id, block_position, block_position_spatial_logits

from trainers.evaluator import ResNetEvaluator
from trainers.trainer import cls_tripletTrainer, cls_tripletTrainer_for_prior_posterior, \
    cls_tripletTrainer_for_prior_posterior_multibranch, cls_tripletTrainer_test
from utils.loss import CrossEntropyLabelSmooth, TripletLoss, Margin, LikelihoodLoss
from utils.LiftedStructure import LiftedStructureLoss
from utils.DistWeightDevianceLoss import DistWeightBinDevianceLoss
from utils.serialization import Logger, save_checkpoint
from utils.transforms import TestTransform, TrainTransform


def train(**kwargs):
    opt._parse(kwargs)

    # set random seed and cudnn benchmark
    torch.manual_seed(opt.seed)
    os.makedirs(opt.save_dir, exist_ok=True)
    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(osp.join(opt.save_dir, 'log_train.txt'))

    print('=========user config==========')
    pprint(opt._state_dict())
    print('============end===============')

    if use_gpu:
        print('currently using GPU')
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(opt.seed)
    else:
        print('currently using cpu')

    print('initializing dataset {}'.format(opt.dataset))
    dataset = data_manager.init_dataset(name=opt.dataset, mode=opt.mode)

    pin_memory = True if use_gpu else False

    summary_writer = SummaryWriter(osp.join(opt.save_dir, 'tensorboard_log'))

    trainloader = DataLoader(
        ImageData(dataset.train, TrainTransform(opt.datatype)),
        sampler=RandomIdentitySampler(dataset.train, opt.num_instances),
        batch_size=opt.train_batch, num_workers=opt.workers,
        pin_memory=pin_memory, drop_last=True
    )

    queryloader = DataLoader(
        ImageData(dataset.query, TestTransform(opt.datatype)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    galleryloader = DataLoader(
        ImageData(dataset.gallery, TestTransform(opt.datatype)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )
    queryFliploader = DataLoader(
        ImageData(dataset.query, TestTransform(opt.datatype, True)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    galleryFliploader = DataLoader(
        ImageData(dataset.gallery, TestTransform(opt.datatype, True)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    print('initializing model ...')
    if opt.model_name == 'softmax' or opt.model_name == 'softmax_triplet':
        model = ResNetBuilder(dataset.num_train_pids, 1, True)
    elif opt.model_name == 'triplet':
        model = ResNetBuilder(None, 1, True)
    elif opt.model_name == 'bfe':
        if opt.datatype == "person":
            model = BFE(dataset.num_train_pids, 1.0, 0.33)
        else:
            model = BFE(dataset.num_train_pids, 0.5, 0.5)
    elif opt.model_name == 'ide':
        model = IDE(dataset.num_train_pids)
    elif opt.model_name == 'resnet':
        model = Resnet(dataset.num_train_pids)

    optim_policy = model.get_optim_policy()

    if opt.pretrained_model:
        state_dict = torch.load(opt.pretrained_model)['state_dict']
        # state_dict = {k: v for k, v in state_dict.items() \
        #        if not ('reduction' in k or 'softmax' in k)}
        model.load_state_dict(state_dict, False)
        print('load pretrained model ' + opt.pretrained_model)
    print('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

    if use_gpu:
        model = nn.DataParallel(model).cuda()
    reid_evaluator = ResNetEvaluator(model)

    if opt.evaluate:
        reid_evaluator.evaluate(queryloader, galleryloader,
                                queryFliploader, galleryFliploader, re_ranking=opt.re_ranking, savefig=opt.savefig)
        return

    # xent_criterion = nn.CrossEntropyLoss()
    xent_criterion = CrossEntropyLabelSmooth(dataset.num_train_pids)
    bce_criterion = nn.BCELoss()
    lkd_criterion = LikelihoodLoss(dataset.num_train_pids)

    if opt.loss == 'triplet':
        embedding_criterion = TripletLoss(opt.margin)
    elif opt.loss == 'lifted':
        embedding_criterion = LiftedStructureLoss(hard_mining=True)
    elif opt.loss == 'weight':
        embedding_criterion = Margin()

    def criterion(triplet_y, softmax_y, labels):
        losses = [embedding_criterion(output, labels)[0] for output in triplet_y] + \
                 [xent_criterion(output, labels) for output in softmax_y]
        loss = sum(losses)
        return loss

    def criterion_for_position(predicted_mask, label_mask):
        loss = bce_criterion(predicted_mask, label_mask)
        return loss

    def criterion_for_position_reg(triplet_y, softmax_y, labels, predicted_mask, label_mask):
        losses = [embedding_criterion(output, labels)[0] for output in triplet_y] + \
                 [xent_criterion(output, labels) for output in softmax_y]
        loss = sum(losses)
        loss += bce_criterion(predicted_mask, label_mask)
        return loss

    def criterion_for_prior_posterior(cls_score, posterior_mu, posterior_sigma, prior_mu, prior_sigma, labels):
        lkd_loss, logits_with_margin = lkd_criterion(cls_score, labels)
        xent_loss = xent_criterion(logits_with_margin, labels)
        triplet_loss = embedding_criterion(posterior_mu, labels)[0]
        loss = lkd_loss + xent_loss + triplet_loss
        return loss, lkd_loss, xent_loss, triplet_loss, posterior_mu, posterior_sigma, prior_mu, prior_sigma

    def criterion_for_prior_posterior_multibranch(cls_score, posterior_mu, posterior_sigma, prior_mu, prior_sigma,
                                                  labels):
        lkd_losses = []
        xent_losses = []
        triplet_losses = []
        losses = []
        for i in range(len(cls_score)):
            lkd_loss, logits_with_margin = lkd_criterion(cls_score[i], labels)
            xent_loss = xent_criterion(logits_with_margin, labels)
            triplet_loss = embedding_criterion(posterior_mu[i], labels)[0]
            loss = lkd_loss + xent_loss + triplet_loss
            losses.append(loss)
            lkd_losses.append(lkd_loss)
            triplet_losses.append(triplet_loss)
            xent_losses.append(xent_loss)
        loss, lkd_loss, triplet_loss, xent_loss = sum(losses), sum(lkd_losses), sum(triplet_losses), sum(xent_losses)
        return loss, lkd_loss, xent_loss, triplet_loss, posterior_mu, posterior_sigma, prior_mu, prior_sigma

    # get optimizer
    if opt.optim == "sgd":
        optimizer = torch.optim.SGD(optim_policy, lr=0., momentum=0.9, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(optim_policy, lr=0., weight_decay=opt.weight_decay)

    start_epoch = opt.start_epoch
    # get trainer and evaluator
    reid_trainer = cls_tripletTrainer_test(opt, model, optimizer, criterion, summary_writer)
    rank1 = reid_trainer.train(queryloader, galleryloader)
    print('Best rank-1 {:.1%}'.format(rank1))



def test(model, queryloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target, _ in queryloader:
            output = model(data).cpu()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    rank1 = 100. * correct / len(queryloader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(queryloader.dataset), rank1))
    return rank1


if __name__ == '__main__':
    import fire
    fire.Fire()
