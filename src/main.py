import argparse
import os
import random
from secrets import choice
import numpy as np
import torch
from dataloader import get_dataloader
from trainer import BaseTrainer, SimTrainer, ClfTrainer

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main():
    parser = argparse.ArgumentParser()
    # customized the dataset and labeld size
    parser.add_argument('--dataset', type=str, default='yahoo')
    parser.add_argument('--num_labeled', type=int, default=200)

    # customize the model
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus')
    parser.add_argument('--num_augs', type=int, default=2, help='the number of augmentation methods used in a single run')
    parser.add_argument('--hidden_size', type=int, default=128, help='the linear hidden dimension in the classifier on the top of bert')
    parser.add_argument('--scorer_hidden_size', type=int, default=768, help='the linear hidden dimension in the classifier on the top of bert')
    
    # task
    parser.add_argument('--task', type=str, default='semi')
    parser.add_argument('--aug_metric', type=str, default='base', choices=['base', 'sim'])
    parser.add_argument('--seed', type=int, default=2022)

    # customize the trainer
    parser.add_argument('--num_epoch', type=int, default=20, help='num epoches to train')
    parser.add_argument('--warmup_epoch', type=int, default=0, help='num epoches to train')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='the optimizer chosen to train the model')
    parser.add_argument('--thr', type=float, default=0.9, help='the threshold chosen to filter "truths" of low probabilities')
    parser.add_argument('--mu', type=int, default=4, help='the batchsize ratio between unlabeled set and labeled set')
    parser.add_argument('--lbd', type=float, default=0.1, help='the coefficient that controls the strength of the backpropagation of unlabled loss')
    parser.add_argument('--patience', type=int, default=5, help='patience in the training process')
    parser.add_argument('--aux_metric', type=str, default='CE', help='the metric for strength comparison of two augmentation predictions (default: cross entropy)')
    parser.add_argument('--method', type=str, default='aaa', help='the metric for strength comparison of two augmentation predictions (default: cross entropy)')
    parser.add_argument('--lr_main', type=float, default=1e-3, help='learning rate for the main task')
    parser.add_argument('--lr_bert', type=float, default=1e-5, help='learning rate for the bert model')
    parser.add_argument('--lr_aux', type=float, default=1e-4, help='learning rate for the auxiliary task')
    
    config = parser.parse_args()
    
    train_loader_l, train_loader_u, dev_loader, test_loader, num_class = get_dataloader(os.path.join('../data', config.dataset), config.num_labeled, config.mu, config.task)
    loaders = {
        'train_loader_l': train_loader_l,
        'train_loader_u': None if config.task == 'baseline' else train_loader_u,
        'dev_loader': dev_loader, 
        'test_loader': test_loader
    }
    if config.dataset == 'imdb':
        config.num_classes = 2
    else:
        config.num_classes = num_class
    set_seed(config)
    
    if config.aug_metric == 'base':
        trainer = ClfTrainer(config, loaders)
    elif config.aug_metric == 'sim':
        trainer = SimTrainer(config, loaders)
        
    if config.task == 'baseline':
        trainer.run_base()
    elif config.task == 'semi':
        trainer.run()

if __name__ == '__main__':
    main()