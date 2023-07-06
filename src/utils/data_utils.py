from typing import Counter
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import torch
from imblearn.under_sampling import *
from typing import *
from copy import deepcopy
 
def construct_balanced_subset(x, y, oversample=False, seed=-1): 
    if oversample:
        ros = RandomOverSampler(random_state=42) 
        x = np.array(x)
        b, c, w, h = x.shape
        x = x.reshape(b, -1)
        X_resampled, y_resampled = ros.fit_resample(x, np.array(y))
        X_resampled = X_resampled.reshape(-1, c, w, h)
        return X_resampled, y_resampled
    else: 
        if seed > 0:
            np.random.seed(seed)
        xdata, ydata = [], []
        minsize = np.inf
        for cls_ in np.unique(y):
            xdata.append(x[y == cls_])
            ydata.append(y[y == cls_])
            if ydata[-1].shape[0] < minsize:
                minsize = ydata[-1].shape[0]
        for i in range(len(xdata)): 
            idx = np.arange(xdata[i].shape[0])
            np.random.shuffle(idx) 
            xdata[i] = xdata[i][idx][:minsize]
            ydata[i] = ydata[i][idx][:minsize]
        # !list
        return np.concatenate(xdata, 0), np.concatenate(ydata, 0)

def not_random_under_sampling(dataloder, model_fn, device):
    # bad acc... strongly bias towards new classes.
    targets, features = [], []
    with torch.no_grad():
        for _inputs, _targets in dataloder:
            _inputs = _inputs.to(device, non_blocking=True)
            _targets = _targets
            _features = model_fn(_inputs)
            features.append(_features)
            targets.append(_targets)
    X, y = torch.cat(features).cpu().numpy(), torch.cat(targets).cpu().numpy()
    # import pdb;pdb.set_trace()
    cc = OneSidedSelection(random_state=0)
    cc.fit_resample(X, y)
    return cc.sample_indices_

def remap_split_targets(y: np.ndarray):
    uni = np.unique(y)
    return np.searchsorted(y, uni)
 