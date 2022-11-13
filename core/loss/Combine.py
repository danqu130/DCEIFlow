import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('core')

from utils.utils import build_module


class Combine(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss_names = args.loss
        self.loss_weights = args.loss_weights
        self.loss_num = len(self.loss_names)
        self.loss = []
        for i in range(self.loss_num):
            self.loss.append(build_module("core.loss", self.loss_names[i])(args))

    def forward(self, output, target):

        loss_all = 0.
        loss_dict = {}
        for i in range(self.loss_num):
            loss_each, loss_metric = self.loss[i](output, target)
            loss_all += loss_each * self.loss_weights[i]
            loss_dict.update(loss_metric)

        loss_dict.update({
            "loss": loss_all,
        })

        return loss_dict
