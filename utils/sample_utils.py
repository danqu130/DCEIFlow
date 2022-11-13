import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate


def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow16(flow, mode='bilinear'):
    new_size = (16 * flow.shape[2], 16 * flow.shape[3])
    return 16 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def upflow4(flow, mode='bilinear'):
    new_size = (4 * flow.shape[2], 4 * flow.shape[3])
    return 4 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def upflow2(flow, mode='bilinear'):
    new_size = (2 * flow.shape[2], 2 * flow.shape[3])
    return 2 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def downflow2(flow, mode='bilinear'):
    new_size = (flow.shape[2] // 2, flow.shape[3] // 2)
    return 0.5 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def downflow4(flow, mode='bilinear'):
    new_size = (flow.shape[2] // 4, flow.shape[3] // 4)
    return 0.25 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def downflow8(flow, mode='bilinear'):
    new_size = (flow.shape[2] // 8, flow.shape[3] // 8)
    return 0.125 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def downflow2_pool2d(flow):
    _, _, h, w = flow.size()
    return F.adaptive_avg_pool2d(flow, [h//2, w//2])


def downgrid2(coord, mode='bilinear'):
    b, _, ht, wd = coord.shape
    coords_0 = coords_grid(b, ht, wd).to(coord.device)
    flow = coord - coords_0
    flow = 0.5 * F.interpolate(flow, size=(ht // 2, wd // 2),
                               mode=mode, align_corners=True)
    coords_1 = coords_grid(b, ht // 2, wd // 2).to(coord.device)
    return coords_1 + flow


def make_coord_center(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret
