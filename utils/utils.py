import os
import random
import torch
import numpy as np
import logging
import importlib
import torch.nn.functional as F
from pytz import timezone
from datetime import datetime


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, div=8, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // div) + 1) * div - self.ht) % div
        pad_wd = (((self.wd // div) + 1) * div - self.wd) % div
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad_batch(self, batch):
        pad_batch = {}
        for key in batch.keys():
            if torch.is_tensor(batch[key]) and len(batch[key].shape) == 4:
                pad_batch[key] = F.pad(batch[key], self._pad, mode='replicate')
            elif torch.is_tensor(batch[key]) and len(batch[key].shape) == 3:
                pad_batch[key] = F.pad(batch[key].unsqueeze(1), self._pad, mode='replicate').squeeze(1)
        return pad_batch

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        if x is not None:
            ht, wd = x.shape[-2:]
            c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
            if len(x.shape) == 4:
                return x[..., c[0]:c[1], c[2]:c[3]]
            elif len(x.shape) == 3:
                return x[:, c[0]:c[1], c[2]:c[3]]
            else:
                raise NotImplementedError('not supported pad size for x.shape {}'.format(x.shape))
        else:
            return None


def ensure_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())


def build_module(module_path, module_name):
    module_path = module_path + '.' + module_name
    try:
        module = importlib.import_module(module_path)
        module = getattr(module, module_name)
    except Exception as e:
        logging.exception(e)
        raise ModuleNotFoundError("No module named '{}'".format(module_path))

    return module
