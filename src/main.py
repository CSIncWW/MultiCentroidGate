import sys
import os
import os.path as osp
import copy
import time
import shutil
import cProfile
import logging
from pathlib import Path
import numpy as np
import random
from easydict import EasyDict as edict
import factory
import json
 
import yaml

from rehearsal.memory_size import MemorySize
import torch

import utils
from pathlib import Path
import datetime
import timm.utils 
from torch.utils.data import DataLoader
import args 
import torch.distributed as dist 
import trainer.seperate
import trainer.metric_re
from utils.distributed import is_main_process

def initialization(): 
    cfg = args.init_cfg() 
    utils.init_distributed_mode(cfg) # it will substitude print. so use print without get_rank()
    utils.set_seed(cfg['seed'], 0)
    # cfg.lr = cfg.lr * cfg.batch_size * dist.get_world_size() / 512.0
    # cfg.lr = cfg.lr * dist.get_world_size()
 
    # create exp
    t = datetime.datetime.now().strftime("%Y%m%d%H%M")
    if cfg.debug:
        exp_folder = Path(f"/tmp")
    else:
        exp_folder = Path(f"exps/{cfg['exp_name']}-{t}/")
    if utils.is_main_process():
        exp_folder.mkdir(parents=True, exist_ok=True)
        (exp_folder / 'ckpt').mkdir(parents=True, exist_ok=True)
        (exp_folder / 'mem').mkdir(parents=True, exist_ok=True)

    # logging
    # logger = utils.make_logger("log", savedir=exp_folder) 
    cfg.exp_folder = exp_folder
    cfg.ckpt_folder = exp_folder / 'ckpt'
    cfg.mem_folder = exp_folder / 'mem'
    if utils.is_main_process(): 
        utils.make_tensorboard(str(exp_folder)) 
    return cfg

def train():
    cfg = initialization() 
    print(cfg)

    start_time = time.time()
    _train(cfg) 
    print("Training finished in {}s.".format(int(time.time() - start_time)))

def _train(cfg):
    inc_dataset = factory.create_data(cfg)
    tb = utils.get_tensorboard()

    model = factory.create_trainer(cfg, inc_dataset) 

    results = {"cfg": utils.del_unjsonable(cfg.__dict__), "results": [], "avg1": "", "avg5": ""}

    for task_i in range(inc_dataset.n_tasks):
        task_info, train_dataset, val_dataset, test_dataset = inc_dataset.new_task()  

        train_loader = factory.create_dataloader(cfg, train_dataset, cfg.distributed, True, drop_last=True)
        val_loader = factory.create_dataloader(cfg, val_dataset, False, False)

        cfg.new_task(task_info["increment"]) 
    
        model.before_task()
        if task_i >= cfg.start_task:
            model.train_task(train_loader, val_loader)
        model.after_task()
 
        if utils.is_main_process():  #and task_i >= cfg.start_task
            print(task_i)
            test_loader = factory.create_dataloader(cfg, test_dataset, False, False) 
            print("Eval on {}->{}.".format(0, task_info["max_class"]))
            tag = "{}->{}.".format(0, task_info["max_class"])
            if isinstance(model, trainer.seperate.IncModel):
                ypred, ytrue = model.eval_task(test_loader)
                acc_stats0 = utils.generate_report(ypred, ytrue, cfg.increments)  
                print(f"top1:{acc_stats0['top1']}")  
                results["results"].append({
                    tag:[acc_stats0['top1'], acc_stats0['top5']]
                })
            elif isinstance(model, trainer.metric_re.IncModel):
                r, ytrue = model.eval_task(test_loader)
                ypred, ypred_gate, gated_ypred, gated_ypred_gcl, gated_ypred_mix = \
                    r['logit'], r['gate_task_logit'], r['final_logit'], r['final_logit_gcl'], r['final_logit_mix']
                acc_stats0 = utils.generate_report(ypred, ytrue, cfg.increments) 
                # acc_stats1 = utils.generate_report(ypred_gate, ytrue, cfg.increments)
                acc_stats1 = utils.generate_report(ypred_gate, utils.target_to_task(ytrue, cfg.increments), [1] * (1 + task_i))
                acc_stats2 = utils.generate_report(gated_ypred, ytrue, cfg.increments) 
                acc_stats3 = utils.generate_report(gated_ypred_gcl, ytrue, cfg.increments) 
                acc_stats4 = utils.generate_report(gated_ypred_mix, ytrue, cfg.increments) 
            
                print(f"top1:{acc_stats0['top1']}") 
                print(f"top1:{acc_stats1['top1']}")  
                print(f"top1:{acc_stats2['top1']}") 
                print(f"top1:{acc_stats3['top1']}") 
                print(f"top1:{acc_stats4['top1']}") 
                results["results"].append({
                    tag:[acc_stats0['top1'], acc_stats1['top1'], acc_stats2['top1'], acc_stats3['top1'], acc_stats4['top1']]
                })
            else:
                ypred, ytrue = model.eval_task(test_loader)
                acc_stats0 = utils.generate_report(ypred, ytrue, cfg.increments)  

                #Logging
                tb.add_scalar(f"taskaccu", acc_stats0["top1"]["total"], task_i)
            
                print(f"top1:{acc_stats0['top1']}") 
                results["results"].append(acc_stats0)
        dist.barrier()

    if utils.is_main_process():
        # top1_avg_acc, top5_avg_acc = utils.compute_avg_inc_acc(results["results"])

        # print("Average Incremental Accuracy Top 1: {} Top 5: {}.".format(
        #     top1_avg_acc,
        #     top5_avg_acc,
        # ))
        # results["avg1"] = top1_avg_acc
        # results["avg5"] = top5_avg_acc 
        with open(cfg.exp_folder / 'results.json', 'w+') as f:
            f.write(utils.to_json(results))


if __name__ == "__main__":
    train()
