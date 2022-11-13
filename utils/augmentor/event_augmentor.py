import sys
sys.path.append('core/utils')
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ColorJitter


def resize_flow(flow, des_height, des_width, method='bilinear'):
    # improper for sparse flow
    src_height = flow.shape[1]
    src_width = flow.shape[2]
    if src_width == des_width and src_height == des_height:
        return flow
    ratio_height = float(des_height) / float(src_height)
    ratio_width = float(des_width) / float(src_width)

    flow = np.transpose(flow, (1, 2, 0))
    if method == 'bilinear':
        flow = cv2.resize(
            flow, (des_width, des_height), interpolation=cv2.INTER_LINEAR)
    elif method == 'nearest':
        flow = cv2.resize(
            flow, (des_width, des_height), interpolation=cv2.INTER_NEAREST)
    else:
        raise Exception('Invalid resize flow method!')
    flow = np.transpose(flow, (2, 0, 1))

    flow[0, :, :] = flow[0, :, :] * ratio_width
    flow[1, :, :] = flow[1, :, :] * ratio_height
    return flow


def horizontal_flip_flow(flow):
    flow = np.transpose(flow, (1, 2, 0))
    flow = np.copy(np.fliplr(flow))
    flow = np.transpose(flow, (2, 0, 1))
    flow[0, :, :] *= -1
    return flow


def vertical_flip_flow(flow):
    flow = np.transpose(flow, (1, 2, 0))
    flow = np.copy(np.flipud(flow))
    flow = np.transpose(flow, (2, 0, 1))
    flow[1, :, :] *= -1
    return flow


def remove_ambiguity_flow(flow_img, err_img, threshold_err=10.0):
    thre_flow = flow_img
    mask_img = np.ones(err_img.shape, dtype=np.uint8)
    mask_img[err_img > threshold_err] = 0.0
    thre_flow[err_img > threshold_err] = 0.0
    return thre_flow, mask_img


class EventFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False, spatial_aug_prob=0.8):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = spatial_aug_prob
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3/3.14)
        self.asymmetric_color_aug_prob = 0.2
        # self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if torch.FloatTensor(1).uniform_(0, 1).item() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(
                Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(
                Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(
                Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def spatial_transform(self, event, img1, img2, flow, flow10=None, occ=None, occ10=None, event_r=None):

        if self.do_flip:
            if torch.FloatTensor(1).uniform_(0, 1).item() < self.h_flip_prob:  # h-flip
                event = event[:, :, ::-1]
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                if flow10 is not None:
                    flow10 = flow10[:, ::-1] * [-1.0, 1.0]
                if occ is not None:
                    occ = occ[:, ::-1]
                if occ10 is not None:
                    occ10 = occ10[:, ::-1]
                if event_r is not None:
                    event_r = event_r[:, :, ::-1]

            if torch.FloatTensor(1).uniform_(0, 1).item() < self.v_flip_prob:  # v-flip
                event = event[:, ::-1, :]
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]
                if flow10 is not None:
                    flow10 = flow10[::-1, :] * [1.0, -1.0]
                if occ is not None:
                    occ = occ[::-1, :]
                if occ10 is not None:
                    occ10 = occ10[::-1, :]
                if event_r is not None:
                    event_r = event_r[:, ::-1, :]

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

        event = event[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        if flow10 is not None:
            flow10 = flow10[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        if occ is not None:
            occ = occ[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        if occ10 is not None:
            occ10 = occ10[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        if event_r is not None: 
            event_r = event_r[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return event, img1, img2, flow, flow10, occ, occ10, event_r

    def __call__(self, event, img1, img2, flow, flow10=None, occ=None, occ10=None, event_r=None):
        img1, img2 = self.color_transform(img1, img2)
        event, img1, img2, flow, flow10, occ, occ10, event_r = self.spatial_transform(\
            event, img1, img2, flow, flow10, occ, occ10, event_r)

        event = np.ascontiguousarray(event)
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        if flow10 is not None:
            flow10 = np.ascontiguousarray(flow10)
        if occ is not None:
            occ = np.ascontiguousarray(occ)
        if occ10 is not None:
            occ10 = np.ascontiguousarray(occ10)
        if event_r is not None: 
            event_r = np.ascontiguousarray(event_r)

        return event, img1, img2, flow, flow10, occ, occ10, event_r


# TODO SparseEventFlowAugmentor
class SparseEventFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False, spatial_aug_prob=0.8):
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = spatial_aug_prob
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3/3.14)
        self.asymmetric_color_aug_prob = 0.2
        # self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if torch.FloatTensor(1).uniform_(0, 1).item() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(
                Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(
                Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(
                Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def spatial_transform(self, event, img1, img2, flow, valid=None, flow10=None, valid10=None):
        if self.do_flip:
            if torch.FloatTensor(1).uniform_(0, 1).item() < self.h_flip_prob:  # h-flip
                event = event[:, :, ::-1]
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                if valid is not None:
                    valid = valid[:, ::-1]
                if flow10 is not None and valid10 is not None:
                    flow10 = flow10[:, ::-1] * [-1.0, 1.0]
                    valid10 = valid10[:, ::-1]

            if torch.FloatTensor(1).uniform_(0, 1).item() < self.v_flip_prob:  # v-flip
                event = event[:, ::-1, :]
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]
                if valid is not None:
                    valid = valid[:, ::-1]
                if flow10 is not None and valid10 is not None:
                    flow10 = flow10[::-1, :] * [1.0, -1.0]
                    valid10 = valid10[:, ::-1]

        if img1.shape[0] == self.crop_size[0] or img1.shape[1] == self.crop_size[1]:
            pass
        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

            event = event[:, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

            if valid is not None:
                valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            if flow10 is not None and valid10 is not None:
                flow10 = flow10[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
                valid10 = valid10[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return event, img1, img2, flow, valid, flow10, valid10

    def __call__(self, event, img1, img2, flow, valid=None, flow10=None, valid10=None):

        img1, img2 = self.color_transform(img1, img2)
        event, img1, img2, flow, valid, flow10, valid10 = self.spatial_transform(\
            event, img1, img2, flow, valid, flow10, valid10)

        event = np.ascontiguousarray(event)
        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        if valid is not None:
            valid = np.ascontiguousarray(valid)
        if flow10 is not None:
            flow10 = np.ascontiguousarray(flow10)
        if valid10 is not None:
            valid10 = np.ascontiguousarray(valid10)

        return event, img1, img2, flow, valid, flow10, valid10
