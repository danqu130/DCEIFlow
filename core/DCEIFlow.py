import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('.')
sys.path.append('core')

from core.decoder.with_event_updater import BasicUpdateBlockNoMask, SmallUpdateBlock
from core.backbone.raft_encoder import BasicEncoder, SmallEncoder
from core.corr.raft_corr import CorrBlock, AlternateCorrBlock
from utils.sample_utils import coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class EIFusion(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 192, 1, padding=0)
        self.conv2 = nn.Conv2d(input_dim, 192, 1, padding=0)
        self.convo = nn.Conv2d(192*2, input_dim, 3, padding=1)

    def forward(self, x1, x2):
        c1 = F.relu(self.conv1(x1))
        c2 = F.relu(self.conv2(x2))
        out = torch.cat([c1, c2], dim=1)
        out = F.relu(self.convo(out))
        return out + x1


class DCEIFlow(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.small = False
        self.dropout = 0
        self.alternate_corr = False

        self.isbi = args.isbi
        self.event_bins = args.event_bins if args.no_event_polarity is True else 2 * args.event_bins

        if self.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        # feature network, context network, and update block
        if self.small:
            self.fnet = SmallEncoder(input_dim=3, output_dim=128, norm_fn='instance', dropout=self.dropout)
            self.cnet = SmallEncoder(input_dim=3, output_dim=hdim+cdim, norm_fn='none', dropout=self.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)
            self.fusion = EIFusion(input_dim=128)
            self.enet = SmallEncoder(input_dim=self.event_bins, output_dim=128, norm_fn='instance', dropout=self.dropout)
        else:
            self.fnet = BasicEncoder(input_dim=3, output_dim=256, norm_fn='instance', dropout=self.dropout) 
            self.cnet = BasicEncoder(input_dim=3, output_dim=hdim+cdim, norm_fn='batch', dropout=self.dropout)
            self.update_block = BasicUpdateBlockNoMask(self.args, hidden_dim=hdim)
            self.fusion = EIFusion(input_dim=256)
            self.enet = BasicEncoder(input_dim=self.event_bins, output_dim=256, norm_fn='instance', dropout=self.dropout)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def forward(self, batch, iters=12, flow_init=None, upsample=True):
        """ Estimate optical flow between pair of frames """

        image1 = batch['image1']
        image1 = 2 * (image1 / 255.0) - 1.0
        image1 = image1.contiguous()

        image2 = None
        if self.training or self.isbi:
            assert 'image2' in batch.keys()
            image2 = batch['image2']
            image2 = 2 * (image2 / 255.0) - 1.0
            image2 = image2.contiguous()

        event_voxel = batch['event_voxel']
        event_voxel = 2 * event_voxel - 1.0
        event_voxel = event_voxel.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        reversed_emap = None
        with autocast(enabled=self.args.mixed_precision):
            emap = self.enet(event_voxel)
            if self.isbi and 'reversed_event_voxel' in batch.keys():
                assert image2 is not None
                fmap1, fmap2 = self.fnet([image1, image2])
                reversed_event_voxel = batch['reversed_event_voxel']
                reversed_event_voxel = 2 * reversed_event_voxel - 1.0
                reversed_event_voxel = reversed_event_voxel.contiguous()
                reversed_emap = self.enet(reversed_event_voxel)
            else:
                reversed_emap = None
                if image2 is None:
                    fmap1 = self.fnet(image1)
                    fmap2 = None
                else:
                    fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        emap = emap.float()
        if fmap2 is not None:
            fmap2 = fmap2.float()

        with autocast(enabled=self.args.mixed_precision):
            pseudo_fmap2 = self.fusion(fmap1, emap)

        corr_fn = CorrBlock(fmap1, pseudo_fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        flow_predictions_bw = []
        flow_up = None
        flow_up_bw = None
        pseudo_fmap1 = None

        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, emap, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if fmap2 is not None and reversed_emap is not None:

            with autocast(enabled=self.args.mixed_precision):
                # pseudo_fmap1 = fmap2 + r_emap
                pseudo_fmap1 = self.fusion(fmap2, reversed_emap)

            if self.alternate_corr:
                corr_fn = AlternateCorrBlock(fmap2, pseudo_fmap1, radius=self.args.corr_radius)
            else:
                corr_fn = CorrBlock(fmap2, pseudo_fmap1, radius=self.args.corr_radius)

            # run the context network
            with autocast(enabled=self.args.mixed_precision):
                cnet = self.cnet(image2)
                net, inp = torch.split(cnet, [hdim, cdim], dim=1)
                net = torch.tanh(net)
                inp = torch.relu(inp)

            coords0, coords1 = self.initialize_flow(image2)

            if flow_init is not None:
                coords1 = coords1 + flow_init

            for itr in range(iters):
                coords1 = coords1.detach()
                corr = corr_fn(coords1) # index correlation volume

                flow = coords1 - coords0
                with autocast(enabled=self.args.mixed_precision):
                    net, up_mask, delta_flow = self.update_block(net, inp, corr, reversed_emap, flow)

                # F(t+1) = F(t) + \Delta(t)
                coords1 = coords1 + delta_flow

                # upsample predictions
                if up_mask is None:
                    flow_up_bw = upflow8(coords1 - coords0)
                else:
                    flow_up_bw = self.upsample_flow(coords1 - coords0, up_mask)
                
                flow_predictions_bw.append(flow_up_bw)

        if self.training:
            batch = dict(
                flow_preds=flow_predictions,
                flow_preds_bw=flow_predictions_bw,
                flow_init=coords1 - coords0,
                flow_final=flow_up,
                flow_final_bw=flow_up_bw,
                fmap2_gt=fmap2,
                fmap2_pseudo=pseudo_fmap2,
                fmap1_gt=fmap1,
                fmap1_pseudo=pseudo_fmap1,
            )
        else:
            batch = dict(
                flow_preds=flow_predictions,
                flow_init=coords1 - coords0,
                flow_final=flow_up,
                flow_final_bw=flow_up_bw,
            )
        return batch
