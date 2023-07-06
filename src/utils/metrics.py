from collections import defaultdict
import numpy as np
import torch
import numbers
import math


class IncConfusionMeter:
    """Maintains a confusion matrix for a given calssification problem.
    The ConfusionMeter constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems:
    for such problems, please use MultiLabelConfusionMeter.
    Args:
        k (int): number of classes in the classification problem
        normalized (boolean): Determines whether or not the confusion matrix
            is normalized or not
    """
    def __init__(self, k, increments, normalized=False):
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.increments = increments
        self.cum_increments = [0] + np.cumsum(self.increments)
        self.k = k
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix of K x K size where K is no of classes
        Args:
            predicted (tensor): Can be an N x K tensor of predicted scores obtained from
                the model for N examples and K classes or an N-tensor of
                integer values between 0 and K-1.
            target (tensor): Can be a N-tensor of integer values assumed to be integer
                values between 0 and K-1 or N x K tensor, where targets are
                assumed to be provided as one-hot vectors
        """
        if isinstance(predicted, torch.Tensor):
            predicted = predicted.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32), minlength=self.k**2)
        assert bincount_2d.size == self.k**2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        conf = self.conf.astype(np.float32)
        new_conf = np.zeros([len(self.increments), len(self.increments) + 2])
        for i in range(len(self.increments)):
            idxs = range(self.cum_increments[i], self.cum_increments[i + 1])
            new_conf[i, 0] = conf[idxs, idxs].sum()
            new_conf[i, 1] = conf[self.cum_increments[i]:self.cum_increments[i + 1],
                                  self.cum_increments[i]:self.cum_increments[i + 1]].sum() - new_conf[i, 0]
            for j in range(len(self.increments)):
                new_conf[i, j + 2] = conf[self.cum_increments[i]:self.cum_increments[i + 1],
                                          self.cum_increments[j]:self.cum_increments[j + 1]].sum()
        conf = new_conf
        if self.normalized:
            return conf / conf[:, 2:].sum(1).clip(min=1e-12)[:, None]
        else:
            return conf


class ClassErrorMeter:
    def __init__(self, topk=[1], accuracy=False):
        super(ClassErrorMeter, self).__init__()
        self.topk = np.sort(topk)
        self.accuracy = accuracy
        self.reset()

    def reset(self):
        self.sum = {v: 0 for v in self.topk}
        self.n = 0

    def add(self, output, target):
        if isinstance(output, np.ndarray):
            output = torch.Tensor(output)
        if isinstance(target, np.ndarray):
            target = torch.Tensor(target)
        topk = self.topk
        maxk = min(int(topk[-1]), output.shape[1])  # seems like Python3 wants int and not np.int64
        no = output.shape[0]

        pred = output.topk(maxk, 1, True, True)[1]
        correct = pred == target.unsqueeze(1).repeat(1, pred.shape[1])
        
        for k in topk:
            self.sum[k] += no - correct[:, 0:k].sum()
        self.n += no

    def value(self, k=-1):
        if k != -1:
            assert k in self.sum.keys(), \
                'invalid k (this k was not provided at construction time)'
            if self.n == 0:
                return float('nan')
            if self.accuracy:
                return (1. - float(self.sum[k]) / self.n) * 100.0
            else:
                return float(self.sum[k]) / self.n * 100.0
        else:
            return [self.value(k_) for k_ in self.topk]

class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
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

class MultiAverageMeter():
    def __init__(self):
        self.values = defaultdict(AverageMeter)
    
    def update(self, name, value):
        self.values[name].update(value)

    @property
    def avg_all(self):
        m = 0
        for v in self.values.values():
            m += v.avg
        if len(self.values) == 0:
            return 0
        return m / len(self.values) 
    
    @property
    def avg_per(self):
        m = {}
        for k, v in self.values.items():
            m[k] = v.avg
        return m
    
    def get(self, name):
        return self.values[name].avg

 
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [(correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size).item() for k in topk]

def backward_transfer(acc_mat):
    T = len(acc_mat)
    v = []
    for i in range(1, T):
        v.append(np.mean([acc_mat[i][-1] - acc_mat[i][j] for j in range(0, i)]))
    return np.mean(v)