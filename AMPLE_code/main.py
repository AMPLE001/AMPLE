import argparse
import logging
import os
import pickle
import sys

os.chdir(sys.path[0])

import numpy as np
import torch
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam

from data_loader.dataset import DataSet
from modules.model import DevignModel
from trainer import train
from utils import tally_param, debug, set_logger

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
import math
from torch.optim.optimizer import Optimizer, required

import torch.optim

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-6,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)

        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        beta2_t = None
        ratio = None
        N_sma_max = None
        N_sma = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                if beta2_t is None:
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    beta1_t = 1 - beta1 ** state['step']
                    if N_sma >= 5:
                        ratio = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / beta1_t

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:                    
                    step_size = group['lr'] * ratio
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    step_size = group['lr'] / beta1_t
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

if __name__ == '__main__':
    torch.manual_seed(10)
    np.random.seed(10)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn'], default='devign')
    parser.add_argument('--dataset', type=str, help='Name of the dataset for experiment.', default='FFmpeg')
    parser.add_argument('--input_dir', type=str, help='Input Directory of the parser', default='dataset/Devign/devign_cpg_c2_2/')
    parser.add_argument('--log_dir', default='devign_FFmpeg.log', type=str)
    parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default='node_features')
    parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default='targets')

    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=100)
    parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=64)
    args = parser.parse_args()

    model_dir = os.path.join('models', args.dataset)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    log_dir = os.path.join(model_dir, args.log_dir)
    set_logger(log_dir)
    
    logging.info('Check up feature_size: %d', args.feature_size)
    if args.feature_size > args.graph_embed_size:
        print('Warning!!! Graph Embed dimension should be at least equal to the feature dimension.\n'
              'Setting graph embedding size to feature size', file=sys.stderr)
        logging.info('Warning!!! Graph Embed dimension should be at least equal to the feature dimension')
        args.graph_embed_size = args.feature_size

    input_dir = args.input_dir
    processed_data_path = os.path.join(input_dir, 'devign.bin')
    logging.info('#' * 100)
    if True and os.path.exists(processed_data_path):
        debug('Reading already processed data from %s!' % processed_data_path)
        dataset = pickle.load(open(processed_data_path, 'rb'))
        logging.info('Reading already processed data from %s!' % processed_data_path)
    else:
        logging.info('Loading the dataset from %s' % input_dir)
        dataset = DataSet(train_src=os.path.join(input_dir, './devign_cpg_v2_1/devign-train-v2.json'),
                          valid_src=os.path.join(input_dir, './devign_cpg_v2_1/devign-valid-v2.json'),
                          test_src=os.path.join(input_dir, './devign_cpg_v2_1/devign-test-v2.json'),
                          batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
                          l_ident=args.label_tag)
        file = open(processed_data_path, 'wb')
        pickle.dump(dataset, file)   
        file.close()
    logging.info('train_dataset: %d; valid_dataset: %d; test_dataset: %d', len(dataset.train_examples), len(dataset.valid_examples), len(dataset.test_examples))
    logging.info("train_batch: %d, valid_batch: %d, test_batch: %d", len(dataset.train_batches), len(dataset.valid_batches), len(dataset.test_batches))
    logging.info('#' * 100)
    
    assert args.feature_size == dataset.feature_size, \
        'Dataset contains different feature vector than argument feature size. ' \
        'Either change the feature vector size in argument, or provide different dataset.'

    logging.info('Check up model_type: ' + args.model_type)
    if args.model_type == 'ggnn':
        model = GGNNSum(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                        num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
    else:
        model = DevignModel(input_dim=dataset.feature_size, output_dim=100,
                            num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)

    debug('Total Parameters : %d' % tally_param(model))
    debug('#' * 100)
    logging.info('Total Parameters : %d' % tally_param(model))
    logging.info('#' * 100)
    model.cuda()
    loss_function = CrossEntropyLoss(weight=torch.from_numpy(np.array([1,1.2])).float(),reduction='sum')
    loss_function.cuda()
    LR = 1e-4

    optim = RAdam(model.parameters(),lr=LR,weight_decay=1e-6) 
    train(model=model, dataset=dataset, epoches=100, dev_every=len(dataset.train_batches),
          loss_function=loss_function, optimizer=optim,
          save_path=model_dir + '/DevignModel', max_patience=100, log_every=5)  

