import torch
from torch import nn


class ChainedLoss(nn.Module):
    def __init__(self, losses: tuple):
        super(ChainedLoss, self).__init__()
        self._loss_funcs = losses

    def forward(self, logits, labels, label_shared=False):
        loss = list()
        device = 'cpu'
        if isinstance(logits, list):
            if not label_shared:
                labels = [labels] * len(logits)
            assert len(logits) == len(self.labels), 'Logit num was not matched with labels num'
            assert len(logits) == len(self._loss_funcs), 'Logit num was not matched with loss function num'
            for idx, logit in enumerate(logits):
                loss.append(self._loss_funcs[idx](logit, labels[idx]))
                device = logit.device
        else:
            for idx in range(len(self._loss_funcs)):
                loss.append(self._loss_funcs[idx](logits, labels))
                device = logits.device
        return torch.Tensor(loss).to(device).sum()
