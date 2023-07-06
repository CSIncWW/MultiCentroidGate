from typing import List, Text
import torch
import torch.nn as nn
import numpy as np
import itertools


def switch_grad(m, requires_grad):
    for p in m:
        p.requires_grad = requires_grad

def freeze_parameters(m, requires_grad=False):
    if m is None:
        return

    if isinstance(m, nn.Parameter):
        m.requires_grad = requires_grad
    else:
        for p in m.parameters():
            p.requires_grad = requires_grad

def extract_features(model, device, loader):
    targets, features = [], []
    with torch.no_grad():
        for _inputs, _targets in loader:
            _inputs = _inputs.to(device, non_blocking=True)
            _targets = _targets
            _features = model(_inputs)
            features.append(_features)
            targets.append(_targets)
    return torch.cat(features).cpu().numpy(), torch.cat(targets).cpu().numpy()

def extends_linear(linear: nn.Linear, out_features):
    l = nn.Linear(linear.in_features, out_features)
    l.weight[:linear.out_features, :] = linear.weight.detach()
    return l

def merge_module_list(module_list):
    tomerge = [m.weight for m in module_list]
    return torch.cat(tomerge, 0)

def remove_component_from_state_dict(state_dict, components: List[str], is_ddp=False): 
    state_dict_copy = state_dict.copy()
    prefix = "module." if is_ddp else "" 
    components = tuple(map(lambda x: f"{prefix}{x}", components)) 
    for k in state_dict.keys():
        if k.startswith(components):
            del state_dict_copy[k]
    return state_dict_copy

def keep_component_from_state_dict(state_dict, components: List[str], is_ddp=False):
    state_dict_copy = state_dict.copy()
    prefix = "module." if is_ddp else "" 
    components = tuple(map(lambda x: f"{prefix}{x}", components)) 
    for k in state_dict.keys():
        if not k.startswith(components):
            del state_dict_copy[k]
    return state_dict_copy

def per_target_mean(features, targets):
    unique_tasks = torch.unique(targets)
    mean_feature = [features[targets == t].mean(0, keepdim=True) for t in unique_tasks]
    return torch.cat(mean_feature, 0), unique_tasks
    # import pdb; pdb.set_trace()
    for t in unique_tasks:
        sub_targets = targets[targets == t]
        sub_features = features[targets == t]
        unique_sub_targets = torch.unique(sub_targets)
        sub_features_mean = [sub_features[sub_targets == st].mean(0, keepdim=True) for st in unique_sub_targets]
        sub_features_mean = torch.cat(sub_features_mean, 0)
        c = centroid[t]
        row_ind, col_ind = linear_sum_assignment(-sim_fn(sub_features_mean, c).detach().cpu().numpy(), )
        c[col_ind] = c[col_ind] * momentum + (1 - momentum) * sub_features_mean[row_ind]

def update_ema_variables(model, ema_model, alpha, global_step):
    # # Use the true average until the exponential average is more correct
    # alpha = min(1 - 1 / (global_step + 1), alpha)
    # # original impl code: parameter doesn't include bn running_mean， so must train() and forward
    # for ema_param, param in zip(ema_model.parameters(), model.parameters()):
    #     ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    with torch.no_grad():
        model_state_dict = model.state_dict()
        ema_model_state_dict = ema_model.state_dict()
        for entry in ema_model_state_dict.keys():
            ema_param = ema_model_state_dict[entry].clone().detach()
            param = model_state_dict[entry].clone().detach()
            new_param = (ema_param * alpha) + (param * (1. - alpha))
            ema_model_state_dict[entry] = new_param
        ema_model.load_state_dict(ema_model_state_dict)

def get_bn_parameters(model: nn.Module):
    bn_parameters = filter(lambda p: isinstance(p, nn.BatchNorm2d), model.modules())
    bn_parameters = map(lambda x: x.parameters(), bn_parameters)
    bn_parameters = itertools.chain(*bn_parameters)

def count_parameters(model: nn.Module):
    return sum(param.numel() for param in model.parameters())
