import copy
import itertools
 
from args import IncrementalConfig

import factory
import torch
import torch.nn.functional as F
import yaml
from torch import nn
import utils
from utils import TaskInfoMixin 
from .ensemble import Model as EnModel
import numpy as np

class Model(nn.Module):
    def __init__(self, cfg: IncrementalConfig):
        super(Model, self).__init__()
        self.cfg: IncrementalConfig = cfg

        self.der = EnModel(cfg) 
        self.cache_repeat = None
 
    def forward(self, x, mode="train"):  
        return self.der(x) 
    
    def param_groups(self):
        return { 
            "classifier": itertools.chain(self.der.param_groups()['classifier']),
            "experts": self.der.parameters(),
            "experts_classifier": self.der.classifier.parameters(),
        }
    
    def set_train(self, mode):
        if mode == "experts":
            self.der.set_train(mode)
            # self.gate.eval() 
        else:
            raise ValueError() 

    def freeze(self, mode="old"):
        utils.switch_grad(self.parameters(), True)
        if mode == "old":
            self.der.freeze(mode)
            # self.gate.freeze(mode) 
        elif mode == "backbone":
            self.der.freeze("backbone")
        elif mode == "experts":
            utils.switch_grad(self.der.parameters(), False)
        else:
            raise ValueError() 
    
    def reset(self, mode):
        # assert mode in ["classifier"]
        if mode == "classifier":
            self.der.reset_classifier()
            # self.gate.reset_classifier()
        elif mode == "experts_classifier":
            self.der.reset("classifier")
        else:
            raise ValueError() 

    def add_classes(self, n_classes): 
        self.der.add_classes(n_classes)
        
    def get_dict(self, i):
        return {
            "der": self.der.get_dict(i),
            "gate": utils.remove_component_from_state_dict(
                        self.state_dict(), ["der"], False)
        }
    
    def set_dict(self, dict, i, load_gate=True):
        self.der.set_dict(dict["der"], i)
        if load_gate:
            self.load_state_dict(dict["gate"], strict=False)
