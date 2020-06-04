# encoding: utf-8
import warnings
import numpy as np


class DefaultConfig(object):
    seed = 0

    # dataset options
    dataset = 'market1501'
    datatype = 'person'
    mode = 'retrieval'
    # optimization options
    loss = 'triplet'
    optim = 'adam'
    max_epoch = 400
    train_batch = 48
    test_batch = 32
    adjust_lr = True
    lr = 0.0001
    gamma = 0.1
    weight_decay = 5e-4
    momentum = 0.9
    random_crop = False
    margin = None
    num_instances = 4
    num_gpu = 1
    evaluate = False
    savefig = None 
    re_ranking = False

    # model options
    block_choice = None #'position_reg' #'bfe_prior_posterior'#'position_reg' #'position', 'position_reg', prior_posterior, bfe_prior_posterior
    model_name = 'bfe'  # triplet, softmax_triplet, bfe, ide
    last_stride = 1
    pretrained_model = None
    #'/media/ivy/research/BDB_raw/batch-dropblock-network/experiment_res/spatial_based_baseline/block_id_gap_first/market/checkpoint_ep400.pth.tar'
    #'/media/ivy/research/BDB_raw/batch-dropblock-network/experiment_res/spatial_based_baseline_gap_first/block_position_spatial_logits/market/checkpoint_ep200.pth.tar'
    #'/media/ivy/research/BDB_raw/batch-dropblock-network/experiment_res/spatial_based_baseline/block_id_gap_first/market/checkpoint_ep400.pth.tar'
    #'/media/ivy/research/BDB_raw/batch-dropblock-network/experiment_res/two_cutmix_shared/market/checkpoint_ep400.pth.tar'
    #'/media/ivy/research/BDB_raw/batch-dropblock-network/experiment_res/spatial_based_baseline/block_position_spatial_logits/market/checkpoint_ep200.pth.tar'
    #'/media/ivy/research/BDB_raw/batch-dropblock-network/experiment_res/spatial_based_baseline/block_position_spatial_logits/market/checkpoint_ep20.pth.tar'
    #'/media/ivy/research/BDB_raw/batch-dropblock-network/experiment_res/spatial_based_baseline/block_id/market/checkpoint_ep400.pth.tar'
    #'/media/ivy/research/BDB_raw/batch-dropblock-network/experiment_res/spatial_based_baseline/block_position/market/checkpoint_ep1.pth.tar'
    #'/media/ivy/research/BDB_raw/batch-dropblock-network/experiment_res/two_cutmix_attentive_channel_cutmix/market/checkpoint_ep600.pth.tar'
    
    # miscs
    print_freq = 1
    eval_step = 50
    save_dir = './experiment_res/two_cutmix_shared/two_cutmix_shared_forward_version.py/market'
    workers = 10
    start_epoch = 0
    best_rank = -np.inf

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
            if 'cls' in self.dataset:
                self.mode='class'
            if 'market' in self.dataset or 'cuhk' in self.dataset or 'duke' in self.dataset:
                self.datatype = 'person'
            elif 'cub' in self.dataset:
                self.datatype = 'cub'
            elif 'car' in self.dataset:
                self.datatype = 'car'
            elif 'clothes' in self.dataset:
                self.datatype = 'clothes'
            elif 'product' in self.dataset:
                self.datatype = 'product'

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items()
                if not k.startswith('_')}

opt = DefaultConfig()
