import logging
from args import IncrementalConfig 
import factory
import numpy as np
import torch
import torch.cuda.amp
import torch.distributed as dist  
from ds.incremental import IncrementalDataset
# from models import der, moenet, network, resnetatt, share
from models.loss import AutoKD, BCEWithLogitTarget
from models.loss.t import DivLoss
from rehearsal.memory_size import MemorySize
from scipy.spatial.distance import cdist
from timm.optim import create_optimizer, create_optimizer_v2
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler
from torch import div, nn
from torch.nn import DataParallel  # , DistributedDataParallel
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from timm.data import Mixup

# from tools import factory, utils
import utils
from utils.model_utils import extract_features

from .utils import build_exemplar, build_exemplar_broadcast, collate_result
from utils import ClassErrorMeter, AverageMeter, MultiAverageMeter, log_dict
from models import *
from collections import defaultdict
from ds.incremental import WeakStrongDataset
from ds.dataset_transform import dataset_transform
import os.path
from pathlib import Path

from torchvision import transforms as T
from models.loss import aux_loss
from timm.scheduler import CosineLRScheduler

# Constants
EPSILON = 1e-8
logger = logging.getLogger() 

class IncModel():
    def __init__(self, cfg: IncrementalConfig, inc_dataset):
        super(IncModel, self).__init__()
        self.cfg = cfg
        # Data
        self._inc_dataset: IncrementalDataset = inc_dataset

        # Optimizer paras
        self._n_epochs = None

        # memory
        self._memory_size = MemorySize(cfg["mem_size_mode"], cfg["memory_size"], cfg["fixed_memory_per_cls"])
        self._coreset_strategy = cfg["coreset_strategy"]

        # Model
        self._network = factory.create_network(cfg)
        self._parallel_network = None
        self._old_model = None

        # training routing
        try:
            global experts_trainer_cfg, gate_trainer_cfg 
            self.et, self.ef = experts_trainer_cfg[self.cfg.subtrainer] 
        except Exception as e:
            print(e)         
    
    def before_task(self):
        dist.barrier()
        self._memory_size.update_nb_classes(self.cfg.nb_seen_classes)
        self._network.add_classes(self.cfg.nb_task_classes)
        if self.cfg.pretrain_model_dir is not None:
            self.load() 
        if self.cfg["syncbn"]:
            self._network = nn.SyncBatchNorm.convert_sync_batchnorm(self._network)
        dist.barrier() 
        self._network = self._network.cuda()
        self._parallel_network = DDP(self._network,
                                    device_ids=[self.cfg.gpu],
                                    output_device=self.cfg.gpu,
                                    find_unused_parameters=True,
                                    broadcast_buffers=True) 
        self._parallel_network = self._parallel_network.cuda()
        dist.barrier() # required.
        
    def train_task(self, train_loader, val_loader):  
        if self.cfg.part_enable[0] == 1: self.train_experts(train_loader)  
        if self.cfg.part_enable[1] == 1 and (self.cfg.idx_task != 0 or self.cfg.force_ft_first): self.ft_experts(train_loader)
    
    def train_experts(self, train_loader):
        self._parallel_network.train()
        self._parallel_network.module.freeze("old")
        self._parallel_network.module.set_train("experts")
        dist.barrier()
        
        scaler = utils.Scaler(self.cfg.amp) 
        parameters = self._parallel_network.module.param_groups()['experts']
        parameters = filter(lambda x: x.requires_grad, parameters)

        optimizer = create_optimizer(self.cfg, parameters) 
        # scheduler, n_epochs = CosineLRScheduler(optimizer, t_initial=self.cfg.epochs,
        #                                 warmup_t=10,
        #                                 warmup_lr_init=1e-5,
        #                                 warmup_prefix=True), self.cfg.epochs
        scheduler, n_epochs = create_scheduler(self.cfg, optimizer) 
        # print(n_epochs, optimizer,scheduler) 

        with tqdm(range(n_epochs), disable=not utils.is_main_process()) as pbar:
            for e in pbar: 
                train_loader.sampler.set_epoch(e)  
                loss = self.et(self.cfg,
                                    self._parallel_network,
                                    self._old_model,
                                    train_loader,
                                    optimizer, scaler) 
                scheduler.step(e)
                if utils.is_main_process():
                    pbar.set_description(f"E {e} expert loss: {loss:.3f}") 
            
    def ft_experts(self, train_loader):
        dataset = self._inc_dataset.get_custom_dataset("train", "train", True)  
        train_loader = factory.create_dataloader(self.cfg, dataset, self.cfg.distributed, True)

        self._network.reset("experts_classifier")
        self._network.freeze("backbone")
        self._parallel_network = DDP(self._network.cuda(),
                                device_ids=[self.cfg.gpu],
                                output_device=self.cfg.gpu,
                                find_unused_parameters=True).cuda()
        dist.barrier()
        self._parallel_network.eval()
        optim = create_optimizer(self.cfg.ft, self._network.param_groups()["experts_classifier"])
        sched, _ = create_scheduler(self.cfg.ft, optim)

        print(self.cfg.ft.epochs)
        scaler = utils.Scaler(self.cfg.amp) 
        with tqdm(range(self.cfg.ft.epochs), disable=not utils.is_main_process()) as pbar:
            for i in pbar:
                train_loader.sampler.set_epoch(i)
                loss = self.ef(
                    self.cfg,
                    self._parallel_network,
                    train_loader,
                    optim,
                    scaler)
                if utils.is_main_process():
                    pbar.set_description(f"Epoch {i} expert finetuning loss {loss: .3f}")
                sched.step(i)

    def after_task(self): 
        self._parallel_network.eval()
        if self.cfg.coreset_feature == "last":
            fn = lambda x: self._network(x)['feature'][:, -512:]
        elif self.cfg.coreset_feature == "all":
            fn = lambda x: self._network(x, "experts")['feature'] 
        
        try:
            idx = np.load(Path(self.cfg.pretrain_model_dir) / f"mem/step{self.cfg.idx_task}.npy")
            self._inc_dataset.data_memory = self._inc_dataset.data_inc[idx]
            self._inc_dataset.targets_memory = self._inc_dataset.targets_inc[idx]
            print("use mem idx cache")
        except Exception as e:
            idx = build_exemplar_broadcast(self.cfg,
                            fn,
                            self._inc_dataset,
                            self._memory_size.mem_per_cls)
            pretrain_has_mem = False 
        if utils.is_main_process() and self.cfg.save_model:
            # new save format... save disk.
            np.save(self.cfg.mem_folder / f"step{self.cfg.idx_task}.npy", idx)
            torch.save(self._network.get_dict(self.cfg.idx_task), 
                        self.cfg.ckpt_folder / f"step{self.cfg.idx_task}.ckpt")

    def eval_task(self, data_loader):
        self._parallel_network.eval()  
        r, t = collate_result(lambda x: self._network(x, "eval"), data_loader)
        return r['logit'], t

global_step = 0
global_kd_step = 0

def forward_experts(cfg: IncrementalConfig, model, old_model, train_loader, optimizer, scaler): 
    meter = MultiAverageMeter()  
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(cfg.gpu, non_blocking=True), targets.to(cfg.gpu, non_blocking=True)
        # inputs, targets = inputs[targets >= cfg.nb_prev_classes], targets[targets >= cfg.nb_prev_classes]
        # oidx = targets >= cfg.nb_prev_classes
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            output = model(inputs, "experts")
            loss = F.cross_entropy(output['logit'], targets)
            loss_aux = torch.zeros([1]).cuda()
            if output['aux_logit'] is not None and cfg.idx_task > 0:
                loss_aux = aux_loss(cfg.aux_cls_type,
                                    cfg.aux_cls_num,
                                    cfg.nb_seen_classes,
                                    output['aux_logit'],
                                    targets,
                                    F.cross_entropy)
            loss_bwd = loss + loss_aux  
        optimizer.zero_grad()
        scaler(loss_bwd, optimizer) 
        meter.update("clf_loss", loss.item())
        meter.update("aux_loss", loss_aux.item())
    if utils.is_main_process(): 
        global global_step 
        log_dict(utils.get_tensorboard(), meter.avg_per, global_step)
        global_step += 1
    return meter.avg_all

def finetune_experts(cfg, model, train_loader, optimizer, scaler): 
    _loss = AverageMeter()  
    for inputs, targets in train_loader:
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)  
        with torch.cuda.amp.autocast(enabled=cfg.amp):
            outputs = model(inputs, "experts")
            loss = F.cross_entropy(outputs['logit'] / cfg.ft.temperature, targets)  
        optimizer.zero_grad()
        scaler(loss, optimizer) 
 
        _loss.update(loss.item())
    return _loss.avg 

experts_trainer_cfg = {
    "baseline": [forward_experts, finetune_experts],
}