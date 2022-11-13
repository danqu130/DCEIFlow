#!/usr/bin/env python
import numpy as np
import math
import cv2

"""
Calculates per pixel flow error between flow_pred and flow_gt. 
event_img is used to mask out any pixels without events (are 0).
If is_car is True, only the top 190 rows of the images will be evaluated to remove the hood of 
the car which does not appear in the GT.
"""
def flow_error_dense(flow_gt, flow_pred, event_img=None, is_car=False):
    max_row = flow_gt.shape[1]

    if event_img is None:
        event_img = np.ones(flow_pred.shape[0:2])
    if is_car:
        max_row = 190

    event_img_cropped = event_img[:max_row, :]
    flow_gt_cropped = flow_gt[:max_row, :, :]

    flow_pred_cropped = flow_pred[:max_row, :, :]

    event_mask = event_img_cropped > 0

    # Only compute error over points that are valid in the GT (not inf or 0).
    flow_mask = np.logical_and(
        np.logical_and(~np.isinf(flow_gt_cropped[:, :, 0]), ~np.isinf(flow_gt_cropped[:, :, 1])),
        np.linalg.norm(flow_gt_cropped, axis=2) > 0)
    total_mask = np.squeeze(np.logical_and(event_mask, flow_mask))

    gt_masked = flow_gt_cropped[total_mask, :]
    pred_masked = flow_pred_cropped[total_mask, :]

    # Average endpoint error.
    EE = np.linalg.norm(gt_masked - pred_masked, axis=-1)
    n_points = EE.shape[0]
    AEE = np.mean(EE)

    # Percentage of points with EE < 3 pixels.
    thresh = 3.
    percent_AEE = float((EE < thresh).sum()) / float(EE.shape[0] + 1e-5)

    return AEE, percent_AEE, n_points

"""
Propagates x_indices and y_indices by their flow, as defined in x_flow, y_flow.
x_mask and y_mask are zeroed out at each pixel where the indices leave the image.
The optional scale_factor will scale the final displacement.
"""
def prop_flow(x_flow, y_flow, x_indices, y_indices, x_mask, y_mask, scale_factor=1.0):
    flow_x_interp = cv2.remap(x_flow,
                              x_indices,
                              y_indices,
                              cv2.INTER_NEAREST)
    
    flow_y_interp = cv2.remap(y_flow,
                              x_indices,
                              y_indices,
                              cv2.INTER_NEAREST)

    x_mask[flow_x_interp == 0] = False
    y_mask[flow_y_interp == 0] = False
        
    x_indices += flow_x_interp * scale_factor
    y_indices += flow_y_interp * scale_factor

    return

"""
The ground truth flow maps are not time synchronized with the grayscale images. Therefore, we
need to propagate the ground truth flow over the time between two images.
This function assumes that the ground truth flow is in terms of pixel displacement, not velocity.

Pseudo code for this process is as follows:

x_orig = range(cols)
y_orig = range(rows)
x_prop = x_orig
y_prop = y_orig
Find all GT flows that fit in [image_timestamp, image_timestamp+image_dt].
for all of these flows:
  x_prop = x_prop + gt_flow_x(x_prop, y_prop)
  y_prop = y_prop + gt_flow_y(x_prop, y_prop)

The final flow, then, is x_prop - x-orig, y_prop - y_orig.
Note that this is flow in terms of pixel displacement, with units of pixels, not pixel velocity.

Inputs:
  x_flow_in, y_flow_in - list of numpy arrays, each array corresponds to per pixel flow at
    each timestamp.
  gt_timestamps - timestamp for each flow array.
  start_time, end_time - gt flow will be estimated between start_time and end time.
"""
def generate_corresponding_gt_flow(flows,
                                   flows_ts,
                                   start_time,
                                   end_time):

    flow_length = len(flows)
    assert flow_length == len(flows_ts) - 1

    x_flow = flows[0][0]
    y_flow = flows[0][1]
    gt_dt = flows_ts[1] - flows_ts[0]
    pre_dt = end_time - start_time

    # if gt_dt > pre_dt:
    if start_time > flows_ts[0] and end_time <= flows_ts[1]:
        x_flow *= pre_dt / gt_dt
        y_flow *= pre_dt / gt_dt
        return np.concatenate((x_flow[np.newaxis, :], y_flow[np.newaxis, :]), axis=0)

    x_indices, y_indices = np.meshgrid(np.arange(x_flow.shape[1]),
                                       np.arange(x_flow.shape[0]))

    x_indices = x_indices.astype(np.float32)
    y_indices = y_indices.astype(np.float32)

    orig_x_indices = np.copy(x_indices)
    orig_y_indices = np.copy(y_indices)

    # Mask keeps track of the points that leave the image, and zeros out the flow afterwards.
    x_mask = np.ones(x_indices.shape, dtype=bool)
    y_mask = np.ones(y_indices.shape, dtype=bool)

    scale_factor = (flows_ts[1] - start_time) / gt_dt
    total_dt = flows_ts[1] - start_time
    
    prop_flow(x_flow, y_flow,
              x_indices, y_indices,
              x_mask, y_mask,
              scale_factor=scale_factor)

    for i in range(1, flow_length-1):
        x_flow = flows[i][0]
        y_flow = flows[i][1]

        prop_flow(x_flow, y_flow,
                  x_indices, y_indices,
                  x_mask, y_mask)
    
        total_dt += flows_ts[i+1] - flows_ts[i]

    gt_dt = flows_ts[flow_length] - flows_ts[flow_length-1]
    pred_dt = end_time - flows_ts[flow_length-1]
    total_dt += pred_dt

    x_flow = flows[flow_length-1][0]
    y_flow = flows[flow_length-1][1]

    scale_factor = pred_dt / gt_dt

    prop_flow(x_flow, y_flow,
              x_indices, y_indices,
              x_mask, y_mask,
              scale_factor)

    x_shift = x_indices - orig_x_indices
    y_shift = y_indices - orig_y_indices
    x_shift[~x_mask] = 0
    y_shift[~y_mask] = 0

    return np.concatenate((x_shift[np.newaxis, :], y_shift[np.newaxis, :]), axis=0)
