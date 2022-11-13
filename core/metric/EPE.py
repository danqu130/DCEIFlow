import torch


def epe(flow_pred, flow_gt, valid_gt=None):

    epe = torch.sum((flow_pred - flow_gt)**2, dim=1).sqrt()
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    epe = epe.view(-1)
    mag = mag.view(-1)

    outlier = (epe > 3.0).float()
    out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()

    if valid_gt is not None:
        val = valid_gt.view(-1) >= 0.5
        metrics = {
            'epe': epe[val].mean(),
            '1px': (epe[val] < 1).float().mean(),
            '3px': (epe[val] < 3).float().mean(),
            '5px': (epe[val] < 5).float().mean(),
            'F1': out[val].mean() * 100,
            'ol': outlier[val].mean() * 100,
        }
    else:
        metrics = {
            'epe': epe.mean(),
            '1px': (epe < 1).float().mean(),
            '3px': (epe < 3).float().mean(),
            '5px': (epe < 5).float().mean(),
            'F1': out.mean() * 100,
            'ol': outlier.mean() * 100,
        }
    return metrics


def epe_f1(flow_pred, flow_gt, valid_gt=None):

    epe = torch.sum((flow_pred - flow_gt)**2, dim=1).sqrt()
    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    epe = epe.view(-1)
    mag = mag.view(-1)

    outlier = (epe > 3.0).float()
    out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()

    if valid_gt is not None:
        val = valid_gt.view(-1) >= 0.5
        metrics = {
            'epe': epe[val].mean(),
            '1px': (epe[val] < 1).float().mean(),
            '3px': (epe[val] < 3).float().mean(),
            '5px': (epe[val] < 5).float().mean(),
            'F1': out[val].mean() * 100,
            'ol': outlier[val].mean() * 100,
        }
    else:
        metrics = {
            'epe': epe.mean(),
            '1px': (epe < 1).float().mean(),
            '3px': (epe < 3).float().mean(),
            '5px': (epe < 5).float().mean(),
            'F1': out.mean() * 100,
            'ol': outlier.mean() * 100,
        }

    return metrics

class EPE:
    def __init__(self, args):
        pass

    def cal(self, flow_pred, flow_gt, flow_valid, event_valid, name):
        if 'mvsec' in name:
            if 'outdoor' in name:
                # remove bottom car 
                # https://github.com/daniilidis-group/EV-FlowNet/blob/master/src/eval_utils.py#L10
                flow_pred = flow_pred[:, :, 0:190, :].contiguous()
                flow_gt = flow_gt[:, :, 0:190, :].contiguous()
                flow_valid = flow_valid[:, :, 0:190, :].contiguous()
                event_valid = event_valid[:, :, 0:190, :].contiguous()

            metric = epe_f1(flow_pred, flow_gt, flow_valid)
            masked_metric = epe_f1(flow_pred, flow_gt, flow_valid * event_valid)

            for key, values in masked_metric.items():
                new_key = "emasked_{}".format(key)
                assert new_key not in metric
                metric[new_key] = values

        else:
            metric = epe(flow_pred, flow_gt)

        return metric

    def __call__(self, output, target, name=None):
        assert name is not None

        flow_pred = output['flow_pred']
        flow_gt = target['flow_gt']
        flow_valid = target['flow_valid']
        if 'event_valid' in target.keys():
            event_valid = target['event_valid']
        else:
            event_valid = None

        metric = self.cal(flow_pred, flow_gt, flow_valid, event_valid, name)

        return metric
