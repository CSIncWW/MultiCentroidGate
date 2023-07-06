from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AutoKD(nn.Module):
    def __init__(self, tau):
        super(AutoKD, self).__init__()
        self.tau = tau

    def forward(self, logit, logit_old):
        logits_for_distill = logit[:, :logit_old.shape[1]]
        _kd_loss = F.kl_div(
            F.log_softmax(logits_for_distill / self.tau, dim=1),
            F.log_softmax(logit_old / self.tau, dim=1),
            reduction='batchmean',
            log_target=True
        ) * (self.tau ** 2)

        lbd = logit_old.shape[1] / logit.shape[1]
        return _kd_loss, lbd  # (1-lbd) * ce +  lbd * kd


class BalancedKD(nn.Module):
    def __init__(self, tau, beta, cls_num_list):
        super().__init__()
        self.tau = tau
        self.beta = beta
        effective_num = 1.0 - np.power(self.beta, cls_num_list)
        per_cls_weights = (1.0 - self.beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / \
            np.sum(per_cls_weights) * len(cls_num_list)

    def forward(self, logit, logit_target): 
        logit = torch.log_softmax(logit/self.tau, dim=1)
        logit_target = torch.softmax(logit_target/self.tau, dim=1) * self.per_cls_weights 
        logit_target = logit_target / logit_target.sum(1)[:, None]
        return -torch.mul(logit_target, logit).sum() / logit.shape[0] 

class MetricKD(nn.Module):
    def __init__(self, method="square"):
        super().__init__()
        self.method = method
    
    def forward(self, feature, feature_T):
        if self.method == "square":
            return self.square(feature, feature_T)
        elif self.method == "cos":
            return self.cos(feature, feature_T)
        elif self.method == "dot":
            return self.dot(feature, feature_T)

    def square(self, feature, feature_T): 
        # don't use amp.
        b = feature.shape[0]
        distT = torch.pow(feature_T, 2).sum(dim=1, keepdim=True).expand(b, b)
        distT = distT + distT.t()
        distT = distT.addmm(feature_T, feature_T.T, beta=1, alpha=-2) # equals to ||a - b||_2
        distT = distT.clamp(min=1e-5).sqrt()

        dist = torch.pow(feature, 2).sum(dim=1, keepdim=True).expand(b, b)
        dist = dist + dist.t()
        dist = dist.addmm(feature, feature.T, beta=1, alpha=-2)  
        dist = dist.clamp(min=1e-5).sqrt()

        return torch.mean(dist-distT) 
    
    def cos(self, feature, feature_T): 
        b, d = feature.shape
        feature = F.normalize(feature, p=2, dim=1) 
        feature_T = F.normalize(feature_T, p=2, dim=1)

        sim = feature @ feature.T
        sim_T = feature_T @ feature_T.T 

        return F.mse_loss(sim, sim_T)
    
    def dot(self, feature, feature_T): 
        b, d = feature.shape
        sim = feature @ feature.T
        sim_T = feature_T @ feature_T.T 

        return F.mse_loss(sim, sim_T) 

class SSIL(nn.Module):
    def __init__(self, tau, nb_tasks, increments) -> None:
        # https://github.com/hongjoon0805/SS-IL-Official/blob/master/trainer/ssil.py
        super().__init__()
        self.tau = tau
        self.nb_tasks = nb_tasks
        self.increments = increments
    
    def forward(self, logit, logit_old):
        # loss_KD
        mid = logit_old.shape[1]
        score = logit[:,:mid].data
        loss_KD = torch.zeros(self.nb_tasks).cuda()

        start_KD = 0
        for t, tn in enumerate(self.increments): 
            end_KD = start_KD + tn 
            soft_target = F.softmax(score[:,start_KD:end_KD] / self.tau, dim=1)
            output_log = F.log_softmax(logit_old[:,start_KD:end_KD] / self.tau, dim=1)
            loss_KD[t] = F.kl_div(output_log, soft_target, reduction='batchmean') * (self.tau**2)
        return loss_KD.sum()