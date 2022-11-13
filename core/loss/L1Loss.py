import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.iters = args.iters
        self.gamma = args.loss_gamma
        self.isbi = args.isbi
        self.max_flow = 400

    def rescaleflow_tosize(self, flow, new_size, mode='bilinear'):
        if new_size[0] == flow.shape[2] and new_size[1] == flow.shape[3]:
            return flow

        h_scale = new_size[0] / flow.shape[2]
        w_scale = new_size[1] / flow.shape[3]
        assert h_scale == w_scale
        return h_scale * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

    def resizeflow_tosize(self, flow, new_size, mode='bilinear'):
        if new_size[0] == flow.shape[2] and new_size[1] == flow.shape[3]:
            return flow

        h_scale = new_size[0] / flow.shape[2]
        w_scale = new_size[1] / flow.shape[3]
        assert h_scale == w_scale
        return F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

    def compute(self, flow_preds, fmap2_gt, fmap2_pseudo, flow_gt, valid_original):

        flow_loss = 0.0
        # exlude invalid pixels and extremely large diplacements
        mag = torch.sum(flow_gt**2, dim=1, keepdim=True).sqrt()
        valid = (valid_original >= 0.5) & (mag < self.max_flow)

        for i in range(len(flow_preds)):
            i_weight = self.gamma**(len(flow_preds) - i - 1)
            if flow_gt.shape == flow_preds[i].shape:
                i_loss = (flow_preds[i] - flow_gt).abs()
                flow_loss += i_weight * (valid * i_loss).mean()
            else:
                scaled_flow_gt = self.resizeflow_tosize(flow_gt, flow_preds[i].shape[2:])
                i_loss = (flow_preds[i] - scaled_flow_gt).abs()
                scaled_mag = torch.sum(scaled_flow_gt**2, dim=1, keepdim=True).sqrt()
                scaled_valid = (self.resizeflow_tosize(valid_original, flow_preds[i].shape[2:]) >= 0.5) & (scaled_mag < self.max_flow)
                flow_loss += i_weight * (scaled_valid * i_loss).mean()

        epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
        epe = epe.view(-1)[valid.view(-1)]
    
        if fmap2_pseudo is not None:
            if isinstance(fmap2_pseudo, list):
                for i in range(len(fmap2_pseudo)):
                    i_weight = self.gamma**(len(fmap2_pseudo) - i - 1) if len(fmap2_pseudo) != 1 else 1.0
                    i_loss = F.l1_loss(fmap2_pseudo[i], fmap2_gt[i]) * 10
                    pseudo_loss += i_weight * i_loss
            else:
                pseudo_loss = F.l1_loss(fmap2_pseudo, fmap2_gt) * 10

            flow_loss += pseudo_loss

        if fmap2_pseudo is None:
            metrics = {
                'l1loss': flow_loss,
                'epe': epe.mean(),
                '1px': (epe < 1).float().mean(),
                '3px': (epe < 3).float().mean(),
                '5px': (epe < 5).float().mean(),
            }
        else:
            metrics = {
                'l1loss': flow_loss,
                'epe': epe.mean(),
                'pseudo': pseudo_loss,
                '1px': (epe < 1).float().mean(),
                '3px': (epe < 3).float().mean(),
                '5px': (epe < 5).float().mean(),
            }

        return flow_loss, metrics


    def forward(self, out, target):
        """ Loss function defined over sequence of flow predictions """
        flow_loss = 0.0

        flow_preds = out['flow_preds']
        fmap2_gt = out['fmap2_gt']
        fmap2_pseudo = out['fmap2_pseudo']
        flow_gt = target['flow_gt']
        valid = target['flow_valid']
        flow_loss_fw, metrics_fw = self.compute(flow_preds, fmap2_gt, fmap2_pseudo, flow_gt, valid)

        if not self.isbi:
            return flow_loss_fw, metrics_fw
        else:
            assert 'flow_preds_bw' in out.keys()
            flow_preds = out['flow_preds_bw']
            fmap2_gt = out['fmap1_gt']
            fmap2_pseudo = out['fmap1_pseudo']
            flow_gt = target['flow10_gt']
            valid = target['flow10_valid']
            flow_loss_bw, metrics_bw = self.compute(flow_preds, fmap2_gt, fmap2_pseudo, flow_gt, valid)

            flow_loss = (flow_loss_fw + flow_loss_bw) * 0.5
            metrics = {}
            for key in metrics_fw:
                assert key in metrics_bw.keys()
                metrics[key] = (metrics_fw[key] + metrics_bw[key]) * 0.5

            return flow_loss, metrics
