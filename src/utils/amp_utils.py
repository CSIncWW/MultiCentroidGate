import torch
import torch.cuda.amp
from timm.utils import dispatch_clip_grad

"""
    optimizer.zero_grad()
    sclaer(loss, optimizer)
"""
class Scaler:
    """a wrapper of timm's scaler. Add some useful option"""
    state_dict_key = "amp_scaler"

    def __init__(self, enable=True):
        self._scaler = torch.cuda.amp.GradScaler(enabled=enable)

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if clip_grad is not None:
            assert parameters is not None
            self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
        self._scaler.step(optimizer)
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
