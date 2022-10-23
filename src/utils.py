import torch
import torch.nn as nn
import torch.nn.functional as F

def compare_and_gen_augchoicemask(preds1, preds2, label, metric='CE'):
    # Current we use cross entropy
    if metric == 'CE':
        metric = nn.CrossEntropyLoss(reduction='none')
    m1, m2 = metric(preds1, label), metric(preds2, label)  # (n, 1)
    all_metrics = torch.stack([m1, m2], dim=-1)
    
    if isinstance(metric, nn.CrossEntropyLoss):
        aug_choice = torch.argmin(all_metrics, dim=-1).long()
    return aug_choice

def move_to_cuda(dict_inputs: dict):
    for k, v in dict_inputs.items():
        dict_inputs[k] = v.cuda()
    
    return dict_inputs

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count