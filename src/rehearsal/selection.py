import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Iterable
from time import time
"""
All dataloader in this model should set shuffle=False. don't use distributed sampler.
"""

def icarl_selection(features, nb_examplars):
    D = features.T
    D = D / (np.linalg.norm(D, axis=0) + 1e-8)
    mu = np.mean(D, axis=1)
    herding_matrix = np.zeros((features.shape[0],))

    w_t = mu
    iter_herding, iter_herding_eff = 0, 0

    while not (
        np.sum(herding_matrix != 0) == min(nb_examplars, features.shape[0])
    ) and iter_herding_eff < 1000:
        tmp_t = np.dot(w_t, D)
        ind_max = np.argmax(tmp_t)
        iter_herding_eff += 1
        if herding_matrix[ind_max] == 0:
            herding_matrix[ind_max] = 1 + iter_herding
            iter_herding += 1

        w_t = w_t + mu - D[:, ind_max]

    herding_matrix[np.where(herding_matrix == 0)[0]] = 10000

    return herding_matrix.argsort()[:nb_examplars]

def d2(model, device, sel_loader, n_classes, task_size, memory_per_class: list):   
    features, targets = [], []
    with torch.set_grad_enabled(False):
        for _x, _y in sel_loader:
            _x = _x.to(device)
            f = model(_x).cpu()
            features.append(f)
            targets.append(_y)
        features = torch.cat(features, 0).numpy()
        targets = torch.cat(targets, 0).numpy()
    idx = [] 
    for class_idx in range(n_classes):
        c_d = np.where(targets == class_idx)[0]
        if class_idx >= n_classes - task_size:
            feat = features[targets == class_idx]
            herding_matrix = icarl_selection(feat, memory_per_class[class_idx])
            idx.append(c_d[herding_matrix])
        else:
            idx.append(c_d[np.arange(memory_per_class[class_idx])])
        # alph = herding_matrix[class_idx]
        # alph = (alph > 0) * (alph < memory_per_class[class_idx] + 1) * 1.0
    return np.concatenate(idx, 0)



def herding_selection(model, device, sel_loader: DataLoader, exemplars_per_class: int) -> Iterable:
    """Selection of new samples. This is based on herding selection, which produces a sorted list of samples of one
    class based on the distance to the mean sample of that class. From iCaRL algorithm 4 and 5:
    https://openaccess.thecvf.com/content_cvpr_2017/papers/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.pdf
    """
    # extract outputs from the model for all train samples
    extracted_features = []
    extracted_targets = []
    with torch.no_grad():
        for images, targets in sel_loader:
            images = images.to(device)
            feats = model(images)
            feats = feats / feats.norm(dim=1).view(-1, 1)  # Feature normalization
            extracted_features.append(feats)
            extracted_targets.extend(targets)
    extracted_features = (torch.cat(extracted_features)).cpu()
    extracted_targets = np.array(extracted_targets)
    result = []
    # iterate through all classes
    for curr_cls in np.unique(extracted_targets):
        # get all indices from current class
        cls_ind = np.where(extracted_targets == curr_cls)[0]
        assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
        assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
        # get all extracted features for current class
        cls_feats = extracted_features[cls_ind]
        # calculate the mean
        cls_mu = cls_feats.mean(0)
        # select the exemplars closer to the mean of each class
        selected = []
        selected_feat = []
        for k in range(exemplars_per_class):
            # fix this to the dimension of the model features
            sum_others = torch.zeros(cls_feats.shape[1])
            for j in selected_feat:
                sum_others += j / (k + 1)  # select mean
            dist_min = np.inf
            # choose the closest to the mean of the current class
            for item in cls_ind:
                if item not in selected:
                    feat = extracted_features[item]
                    dist = torch.norm(cls_mu - feat / (k + 1) - sum_others)  # proto of remain - current.
                    if dist < dist_min:
                        dist_min = dist
                        newone = item
                        newonefeat = feat
            selected_feat.append(newonefeat)
            selected.append(newone)
        result.extend(selected)
    # import pdb; pdb.set_trace()
    return np.array(result)

def entropy_selection(model, device, sel_loader: DataLoader, exemplars_per_class: int) -> Iterable:
    # extract outputs from the model for all train samples
    extracted_logits = []
    extracted_targets = []
    with torch.no_grad():
        for images, targets in sel_loader:
            extracted_logits.append(torch.cat(model(images.to(device)), dim=1))
            extracted_targets.extend(targets)
    extracted_logits = (torch.cat(extracted_logits)).cpu()
    extracted_targets = np.array(extracted_targets)
    result = []
    # iterate through all classes
    for curr_cls in np.unique(extracted_targets):
        # get all indices from current class
        cls_ind = np.where(extracted_targets == curr_cls)[0]
        assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
        assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
        # get all extracted features for current class
        cls_logits = extracted_logits[cls_ind]
        # select the exemplars with higher entropy (lower: -entropy)
        probs = torch.softmax(cls_logits, dim=1)
        log_probs = torch.log(probs)
        minus_entropy = (probs * log_probs).sum(1)  # change sign of this variable for inverse order
        selected = cls_ind[minus_entropy.sort()[1][:exemplars_per_class]]
        result.extend(selected)
    return np.array(result)

def distance_selection(model, device, sel_loader: DataLoader, exemplars_per_class: int) -> Iterable:
    """Selection of new samples. This is based on distance-based selection, which produces a sorted list of samples of
    one class based on closeness to decision boundary of each sample. From RWalk
    http://arxiv-export-lb.library.cornell.edu/pdf/1801.10112
    """
    # extract outputs from the model for all train samples
    extracted_logits = []
    extracted_targets = []
    with torch.no_grad():
        for images, targets in sel_loader:
            extracted_logits.append(torch.cat(model(images.to(device)), dim=1))
            extracted_targets.extend(targets)
    extracted_logits = (torch.cat(extracted_logits)).cpu()
    extracted_targets = np.array(extracted_targets)
    result = []
    # iterate through all classes
    for curr_cls in np.unique(extracted_targets):
        # get all indices from current class
        cls_ind = np.where(extracted_targets == curr_cls)[0]
        assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
        assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
        # get all extracted features for current class
        cls_logits = extracted_logits[cls_ind]
        # select the exemplars closer to boundary
        distance = cls_logits[:, curr_cls]  # change sign of this variable for inverse order
        selected = cls_ind[distance.sort()[1][:exemplars_per_class]]
        result.extend(selected)
    return np.array(result)
