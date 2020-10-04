# encoding: utf-8
import os
import sys
from os import path as osp
from pprint import pprint
import math

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from config import opt
from datasets import data_manager
from datasets.data_loader import ImageData
from datasets.samplers import RandomIdentitySampler
from models.icassp_grad_cam_guided_softmask_two_branch import ResNetBuilder, IDE, Resnet, BFE

# G_branch, G_branch_triplet_KL_xent


# bfe, bfe_channel_drop, two_cutmix, bfe_part_channel_drop, two_cutmix_with_channel_drop, two_cutmix_with_attentive_channel_cutmix
# two_cutmix_shared, block_id, block_position, block_position_spatial_logits

from trainers.evaluator import ResNetEvaluator
from trainers.evaluater_pp import ResNetEvaluator as ResNetEvaluator_pp
from trainers.trainer import cls_tripletTrainer, cls_tripletTrainer_for_prior_posterior, \
    cls_tripletTrainer_for_prior_posterior_multibranch, cls_tripletTrainer_gumble, cls_tripletTrainer_for_lgm
from utils.loss import CrossEntropyLabelSmooth, TripletLoss, Margin, LikelihoodLoss, TripletLoss_cross_attention, TripletLoss_pp
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
    if opt.block_choice == 'prior_posterior':
        reid_evaluator = ResNetEvaluator_pp(model)
    else:
        reid_evaluator = ResNetEvaluator(model)

    if opt.evaluate:
        reid_evaluator.evaluate(queryloader, galleryloader,
                                queryFliploader, galleryFliploader, re_ranking=opt.re_ranking, savefig=opt.savefig)
        return

    # xent_criterion = nn.CrossEntropyLoss()
    xent_criterion = CrossEntropyLabelSmooth(dataset.num_train_pids)
    bce_criterion = nn.BCELoss()
    lkd_criterion = LikelihoodLoss(dataset.num_train_pids)
    cross_attention_triplet = TripletLoss_cross_attention(opt.margin)

    if opt.loss == 'triplet':
        if opt.block_choice=='prior_posterior':
            embedding_criterion = TripletLoss_pp(opt.margin)
        else:
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

    def criterion_for_cross_attention(triplet_y, softmax_y, score_tensors, labels):
        losses = [cross_attention_triplet(triplet_y[i], score_tensors[i], labels)[0] for i in range(len(triplet_y))] + \
                 [xent_criterion(output, labels) for output in softmax_y]
        loss = sum(losses)
        return loss

    def criterion_for_position(predicted_mask, label_mask):
        # loss = bce_criterion(predicted_mask, label_mask)
        loss = nn.functional.kl_div(label_mask.log(), predicted_mask, reduction='sum')
        return loss

    def criterion_for_position_reg(triplet_y, softmax_y, labels, predicted_mask, label_mask):
        losses = [embedding_criterion(output, labels)[0] for output in triplet_y] + \
                 [xent_criterion(output, labels) for output in softmax_y]
        loss = sum(losses)
        loss += xent_criterion(predicted_mask, label_mask)
        # loss += nn.functional.kl_div(label_mask.log(), predicted_mask, reduction='sum')
        return loss

    def criterion_for_prior_posterior(cls_score, posterior_mu, posterior_sigma, prior_mu, prior_sigma, labels):
        lkd_loss, logits_with_margin = lkd_criterion(cls_score, labels)
        xent_loss = xent_criterion(logits_with_margin, labels)
        triplet_loss = embedding_criterion(torch.cat([posterior_mu, posterior_sigma], -1), labels)[0]

        sigma_avg = torch.tensor(5.0)
        thres = torch.log(sigma_avg)
        thres += (1.0 + torch.log(2 * torch.tensor(math.pi))) / 2.0
        ent = torch.mean(F.relu(thres - torch.mean(torch.log(posterior_sigma), -1)))

        loss = lkd_loss + xent_loss + triplet_loss + 0.1 * ent
        return loss, lkd_loss, xent_loss, triplet_loss, posterior_mu, posterior_sigma, prior_mu, prior_sigma

    def criterion_for_lgm(cls_score, posterior_mu, prior_mu, prior_sigma, labels):
        lkd_loss, logits_with_margin = lkd_criterion(cls_score, labels)
        xent_loss = xent_criterion(logits_with_margin, labels)
        triplet_loss = embedding_criterion(posterior_mu, labels)[0]
        loss = lkd_loss + xent_loss + triplet_loss
        return loss, lkd_loss, xent_loss, triplet_loss, posterior_mu, prior_mu, prior_sigma

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
        optimizer = torch.optim.SGD(optim_policy, lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(optim_policy, lr=opt.lr, weight_decay=opt.weight_decay)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, optim_policy), lr=opt.lr, weight_decay=opt.weight_decay)

    start_epoch = opt.start_epoch
    # get trainer and evaluator
    if opt.block_choice == 'position':
        reid_trainer = cls_tripletTrainer(opt, model, optimizer, criterion_for_position, summary_writer)
    elif opt.block_choice == 'position_reg':
        reid_trainer = cls_tripletTrainer(opt, model, optimizer, criterion_for_position_reg, summary_writer)
    elif opt.block_choice == 'prior_posterior':
        reid_trainer = cls_tripletTrainer_for_prior_posterior(opt, model, optimizer, criterion_for_prior_posterior,
                                                              summary_writer)
    elif opt.block_choice == 'lgm':
        reid_trainer = cls_tripletTrainer_for_lgm(opt, model, optimizer, criterion_for_lgm,
                                                              summary_writer)
    elif opt.block_choice == 'bfe_prior_posterior':
        reid_trainer = cls_tripletTrainer_for_prior_posterior_multibranch(opt, model, optimizer,
                                                                          criterion_for_prior_posterior_multibranch,
                                                                          summary_writer)
    elif opt.block_choice == 'cross_attention':
        reid_trainer = cls_tripletTrainer(opt, model, optimizer, criterion_for_cross_attention, summary_writer)
    elif opt.block_choice == 'gumble':
        reid_trainer = cls_tripletTrainer_gumble(opt, model, optimizer, criterion, summary_writer)
    else:
        reid_trainer = cls_tripletTrainer(opt, model, optimizer, criterion, summary_writer)


    # def adjust_lr(optimizer, ep):
    #     if ep < 50:
    #         lr = 1e-4*(ep//5+1)
    #     elif ep < 200:
    #         lr = 1e-3
    #     elif ep < 300:
    #         lr = 1e-4
    #     else:
    #         lr = 1e-5
    #     for p in optimizer.param_groups:
    #         p['lr'] = lr


    def adjust_lr(optimizer, ep):
        if ep < 50:
            lr = 1e-4 * (ep // 5 + 1)
        elif ep < 200:
            lr = 1e-3
        elif ep < 300:
            lr = 1e-4
        elif ep < 500:
            lr = 1e-5
        # else:
        #     lr = 1e-6 / 2  # 5e-6
        # for p in optimizer.param_groups:
        #     p['lr'] = lr
        #     if p['weight_decay'] == 0.:
        #         p['lr'] = lr * 0.01

    def adjust_lr_finetune(optimizer, ep):
        if ep < 200:
            lr = 1e-4
        elif ep < 300:
            lr = 1e-5
        elif ep < 500:
            lr = 1e-6
        else:
            lr = 1e-6 / 2  # 5e-6
        for p in optimizer.param_groups:
            p['lr'] = lr

    # start training
    best_rank1 = opt.best_rank
    best_epoch = 0
    for epoch in range(start_epoch, opt.max_epoch):
        if opt.adjust_lr:
            # print('adjusting learning rate...')
            adjust_lr(optimizer, epoch + 1)
            # adjust_lr_finetune(optimizer, epoch + 1)
        reid_trainer.train(epoch, trainloader)

        # skip if not save model
        if opt.eval_step > 0 and (epoch + 1) % opt.eval_step == 0 or (epoch + 1) == opt.max_epoch:
            if opt.mode == 'class':
                rank1 = test(model, queryloader)
            else:
                rank1 = reid_evaluator.evaluate(queryloader, galleryloader, queryFliploader, galleryFliploader)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({'state_dict': state_dict, 'epoch': epoch + 1},
                            is_best=is_best, save_dir=opt.save_dir,
                            filename='checkpoint_ep' + str(epoch + 1) + '.pth.tar')

    print('Best rank-1 {:.1%}, achived at epoch {}'.format(best_rank1, best_epoch))


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
