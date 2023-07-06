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

class Model( nn.Module,):
    """looks like der but with self classifier"""
    def __init__(self, cfg: IncrementalConfig):
        super(Model, self).__init__()
        self.cfg = cfg 

        self.convnets = nn.ModuleList()
        self.classifier = nn.ModuleList()

        self.remove_last_relu = False
        self.use_bias = False 

        self.aux_classifier = None
        self.out_dim = None
        self.init = "kaiming"
 
    def forward(self, x, method=""): 
        if method == "last":
            lf = self.convnets[-1](x)
            return {
                'logit': self.classifier[-1](lf),
                'aux_logit': self.aux_classifier(lf) if self.aux_classifier is not None else None 
            }
        fl = [conv(x) for conv in self.convnets]
        l = [head(fl[i]) for i, head in enumerate(self.classifier)] 
        l = torch.cat(l, 1)

        return {
            "logit": l,
            "feature": torch.cat(fl, 1),
            "aux_logit": self.aux_classifier(fl[-1]) if self.aux_classifier is not None else None
        } 
            
    def set_train(self, mode):
        self.convnets[:-1].eval()
        self.convnets[-1].train() 
    
    def param_groups(self):
        return { 
            "classifier": self.classifier.parameters(),
        }

    def freeze(self, mode="old"):
        # assert mode in ["old"]
        utils.switch_grad(self.parameters(), True)
        if mode == "old":
            utils.switch_grad(self.convnets[:-1].parameters(), False)
            # utils.switch_grad(self.classifier[:-1].parameters(), False)
        elif mode == "backbone":
            utils.switch_grad(self.convnets.parameters(), False)
        else:
            raise ValueError()
    
    def reset(self, mode):
        assert mode in ["classifier"]
        if mode == "classifier":
            for i in self.classifier:
                # nn.init.kaiming_normal_(i.weight, nonlinearity="linear")
                i.reset_parameters()

    def reset_classifier(self):
        self.reset("classifier")

    def add_classes(self, n_classes): 
        self._add_classes_multi_fc(n_classes)

    def _add_classes_multi_fc(self, n_classes):
        new_clf, out_dim = factory.create_convnet(self.cfg)
        self.out_dim = out_dim

        if self.cfg.idx_task > 0:
            new_clf.load_state_dict(self.convnets[-1].state_dict())
            if self.cfg.aux_cls_type == "1-n":
                self.aux_classifier = self._gen_classifier(self.out_dim,
                                             self.cfg.nb_task_classes + self.cfg.aux_cls_num) 
            elif self.cfg.aux_cls_type == "n-n":
                self.aux_classifier = self._gen_classifier(self.out_dim, self.cfg.nb_seen_classes)

        self.convnets.append(new_clf)
        self.classifier.append(self._gen_classifier(self.out_dim, self.cfg.nb_task_classes)) 

    def _gen_classifier(self, in_features, n_classes):
        classifier = nn.Linear(in_features, n_classes, bias=self.use_bias)
        if self.init == "kaiming": 
            pass
        if self.use_bias:
            nn.init.constant_(classifier.bias, 0.0)

        return classifier
    
    def get_dict(self, i):
        return {
            "fe": self.convnets[i].state_dict(),
            "fc": self.classifier.state_dict()
        }
 
    def set_dict(self, state_dict, i):
        self.convnets[i].load_state_dict(state_dict["fe"])
        self.classifier.load_state_dict(state_dict["fc"])