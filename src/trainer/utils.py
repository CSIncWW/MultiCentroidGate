from collections import Counter, defaultdict
import torch.nn.functional as F
import torch.cuda.amp
from args import IncrementalConfig
from ds.incremental import DummyDataset
import factory
import numpy as np
import utils
import torch
import torch.distributed as dist
from timm.data import Mixup

from utils.data_utils import not_random_under_sampling

def build_exemplar(cfg: IncrementalConfig, model, inc_dataset, memory_per_class: list): 
    # sync = []
    # if utils.is_main_process():
    from rehearsal.selection import d2
    dataset = inc_dataset.get_custom_dataset("train", "test") 
    train_loader = factory.create_dataloader(cfg, dataset, False, False)

    if cfg.coreset_strategy == "iCaRL":
        idx = d2(model,
                    cfg.gpu, train_loader,
                    cfg.nb_seen_classes,
                    cfg.nb_task_classes,
                    memory_per_class)
    else:
        idx = np.arange(len(inc_dataset.data_inc)) 
    #     sync.append(torch.from_numpy(idx).cuda())
    #         # sync.append(torch.zeros([1]).cuda())
    # else:
    #     sync.append(None)
    # dist.barrier()
    # dist.broadcast_object_list(sync, 0)
    # idx = sync[0].cpu().numpy()
    # dist.barrier()
    # return
    # if dist.get_rank() == 0:
    #     # Assumes world_size of 3.
    #     objects = ["foo", 12, {1: 2}] # any picklable object
    # else:
    #     objects = [None, None, None]
    """
        if you want the selected exemplar same in different card. Do following things:
        1. if main process has extra code to execute, keep an eye on random(), Dataloader(generator args).
        2. if you cannot obey the above. set transform to deterministic and shuffle to false
        3. if change DDP or something else, call barrier().
        4. set ddp broad_cast_buffer to True, avoiding bn mean var&conv inconsistency. 
    """
    # if cfg.distributed: 
    #     tidx = torch.from_numpy(idx).cuda()
    #     if not utils.cross_device_equal(tidx):
    #         # np.save(cfg.exp_folder / f"targets_task{cfg.idx_task}_gpu{utils.get_rank()}.npy", inc_dataset.targets_inc)
    #         # np.save(cfg.exp_folder / f"idx_task{cfg.idx_task}_gpu{utils.get_rank()}.npy", idx)
    #         raise Exception("Wrong!!!! different selected idx cross devices") 
    inc_dataset.data_memory = inc_dataset.data_inc[idx]
    inc_dataset.targets_memory = inc_dataset.targets_inc[idx]
    return idx

def build_exemplar_broadcast(cfg: IncrementalConfig, model, inc_dataset, memory_per_class: list): 
    sync = []
    if cfg.coreset_strategy == "disable":
        inc_dataset.data_memory = inc_dataset.data_inc[[]]
        inc_dataset.targets_memory = inc_dataset.targets_inc[[]]
        return []
    if utils.is_main_process():
        from rehearsal.selection import d2
        dataset = inc_dataset.get_custom_dataset("train", "test") 
        train_loader = factory.create_dataloader(cfg, dataset, False, False)

        if cfg.coreset_strategy == "iCaRL":
            idx = d2(model,
                        cfg.gpu, train_loader,
                        cfg.nb_seen_classes,
                        cfg.nb_task_classes,
                        memory_per_class)
        elif cfg.coreset_strategy == "keepall":
            idx = np.arange(len(inc_dataset.data_inc))
        elif cfg.coreset_strategy == "disable":
            idx = []
        sync.append(torch.from_numpy(idx).cuda()) 
    else:
        sync.append(None)
    dist.barrier()
    dist.broadcast_object_list(sync, 0)
    idx = sync[0].cpu().numpy()
    dist.barrier()
    # return
    # if dist.get_rank() == 0:
    #     # Assumes world_size of 3.
    #     objects = ["foo", 12, {1: 2}] # any picklable object
    # else:
    #     objects = [None, None, None]
    """
        if you want the selected exemplar same in different card. Do following things:
        1. if main process has extra code to execute, keep an eye on random(), Dataloader(generator args).
        2. if you cannot obey the above. set transform to deterministic and shuffle to false
        3. if change DDP or something else, call barrier().
        4. set ddp broad_cast_buffer to True, avoiding bn mean var&conv inconsistency. 
    """
    # if cfg.distributed: 
    #     tidx = torch.from_numpy(idx).cuda()
    #     if not utils.cross_device_equal(tidx):
    #         # np.save(cfg.exp_folder / f"targets_task{cfg.idx_task}_gpu{utils.get_rank()}.npy", inc_dataset.targets_inc)
    #         # np.save(cfg.exp_folder / f"idx_task{cfg.idx_task}_gpu{utils.get_rank()}.npy", idx)
    #         raise Exception("Wrong!!!! different selected idx cross devices") 
    inc_dataset.data_memory = inc_dataset.data_inc[idx]
    inc_dataset.targets_memory = inc_dataset.targets_inc[idx]
    return idx

def extract_all(network, data_loader): 
    preds, targets = [], []
    with torch.no_grad():
        for i, (inputs, lbls) in enumerate(data_loader):
            inputs = inputs.cuda(non_blocking=True)
            _preds = network(inputs)['logit'] 
            preds.append(_preds.detach().cpu()) 
            targets.append(lbls.long().cpu())
    preds = torch.cat(preds, axis=0) 
    targets = torch.cat(targets, axis=0)
    return preds, targets

def extract_multiple(fns, data_loader):
    result_list, targets = [[] for _ in range(len(fns))], []
    with torch.no_grad():
        for i, (inputs, lbls) in enumerate(data_loader):
            inputs = inputs.cuda(non_blocking=True)
            for i, fn in enumerate(fns):
                t = fn(inputs)
                result_list[i].append(t.cpu()) 
            targets.append(lbls.long().cpu())
    for i, v in enumerate(result_list):
        result_list[i] = torch.cat(result_list[i], 0)
    targets = torch.cat(targets)
    return result_list, targets

# def extract_

def collate_result(fn, data_loader):
    result = defaultdict(list)
    targets = []
    with torch.no_grad():
        for i, (inputs, lbls) in enumerate(data_loader):
            inputs = inputs.cuda(non_blocking=True)
            r = fn(inputs)
            for k, v in r.items():
                if v is not None:
                    result[k].append(v.cpu()) 
            targets.append(lbls.long().cpu())
    for k, v in result.items():
        result[k] = torch.cat(v)
    targets = torch.cat(targets)
    return result, targets
