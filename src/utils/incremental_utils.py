from copy import deepcopy
import torch
from typing import *
import numpy as np
from .metrics import accuracy

class TaskInfoMixin(object):
    def __init__(self):
        super(TaskInfoMixin, self).__init__()
        self._increments = []
        self._idx_task = -1
        self._nb_seen_classes = 0
        self._nb_task_classes = 0
    
    @property
    def _nb_tasks(self):
        return len(self._increments)

    def new_task(self, nb_task_classes):
        self._idx_task += 1
        self._nb_task_classes = nb_task_classes
        self._increments.append(nb_task_classes)
        self._nb_seen_classes += nb_task_classes

def target_to_task(targets, task_size:List[int]):
    targets = deepcopy(targets)
    prev = 0 
    for i, size in enumerate(task_size): 
        targets[(targets >= prev) & (targets < prev + size)] = i
        prev += size
    return targets

def opt_by_task(logits: torch.tensor, task_size: List[int], opt="sum"):
    sl = []
    prev = 0 
    for i, size in enumerate(task_size):
        if opt == "sum":
            l = logits[:, prev: prev + size].sum(1, keepdims=True)
        elif opt == "amax":
            l = logits[:, prev: prev + size].amax(1, keepdims=True)
        sl.append(l)
        prev += size
    return torch.cat(sl, dim=1)

def per_task_accuracy(pred: torch.tensor, true, task_size: List[int]):
    r = []
    start, end = 0, 0
    for i in range(len(task_size)):
        start = end
        end += task_size[i]
        idxes = torch.where(torch.logical_and(true >= start, true < end))[0]
        top1, = accuracy(pred[idxes], true[idxes], (1,))
        r.append(top1)
    return r

def per_task_norm(feature, nb_tasks):
    norm = torch.chunk(feature, nb_tasks, 1)
    l = []
    for n in norm:
        l.append(torch.norm(n, 2, 1, keepdim=True))
    return torch.cat(l, 1)

def decode_targets(targets: np.ndarray, increments: List[int], overlap: int):
    copy_y = deepcopy(targets) # non decoded.
    task = target_to_task(copy_y, increments)
    ut = np.unique(task)
    for t in ut: copy_y[task == t] -= t * overlap
    return copy_y