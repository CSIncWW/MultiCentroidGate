from .distillation import *
from .t import *


class BCEWithLogitTarget(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, F.one_hot(targets, self.num_classes).float())