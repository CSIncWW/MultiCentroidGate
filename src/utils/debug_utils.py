import torch
import numpy as np
import torch.distributed as dist
import itertools

def print_network_gradient(model):
    for name, parms in model.named_parameters():
        print('-->name:', name,
            '-->grad_requirs:', parms.requires_grad, 
            '--weight', torch.mean(parms.data),
            '-->grad_value:', torch.mean(parms.grad)) 

def find_no_grad(model):
    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)

def check_loss(loss):
    return not bool(torch.isnan(loss).item()) and bool((loss >= 0.0).item())

def draw_lr(epoch, optimizer, scheduler, filename="lr.jpg"):
    lr = []
    for e in range(epoch):
        lr.append(optimizer.param_groups[0]['lr'])
        scheduler.step(e)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(lr)
    fig.savefig(filename)

def cross_device_equal(tensor):
    gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.gather(tensor, gather_t if dist.get_rank() == 0 else None) 
    return all(torch.all(x == gather_t[0]) for x in gather_t)