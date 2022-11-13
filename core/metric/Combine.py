import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('core')

from utils.utils import build_module


class Combine:
    def __init__(self, args):
        super().__init__()
        self.metric_names = args.metric
        self.metric_num = len(self.metric_names)
        self.metrics = []
        for i in range(self.metric_num):
            if self.metric_names[i] == 'epe':
                self.metrics.append(build_module("core.metric", "EPE")(args))
            else:
                self.metrics.append(build_module("core.metric", self.metric_names[i])(args))
        self.all_metrics = {}

    def clear(self):
        self.all_metrics = {}

    def calculate(self, output, target, name=None):
        metrics = {}
        for i in range(self.metric_num):
            metric_each = self.metrics[i](output, target, name)
            metrics.update(metric_each)
        return metrics

    def push(self, metric_each):
        for key in metric_each:
            if key not in self.all_metrics.keys():
                self.all_metrics[key] = []
            self.all_metrics[key].append(metric_each[key])
        return self.all_metrics

    def get_all(self):
        return self.all_metrics

    def summary(self):

        metrics_summary = {}
        metrics_str = ""
        for key, values in self.all_metrics.items():
            num = sum(values) / len(values)
            metrics_summary[key] = num
            metrics_str += "{}:{:8.6f},".format(key, num)
        self.clear()
        return metrics_str, metrics_summary
