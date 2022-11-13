import os
os.environ["KMP_BLOCKTIME"] = "0"
import sys
sys.path.append('utils')
sys.path.append('utils/datasets')
sys.path.append('utils/augmentor')
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import numpy as np
from glob import glob
import torch
import torch.utils.data.dataset as dataset

from utils.event_uitls import eventsToVoxel
from utils.file_io import read_gen, readDenseFlow, read_event_h5
from utils.augmentor import fetch_augmentor


FlyingChairs2_BAD_ID = [
    '0000114',
    '0000163',
    '0000491',
    '0000621',
    '0000107',
    '0011516',
    '0011949',
    '0019593',
    '0013451',
    '0006500',
    '0019693',
    '0009912',
    '0016755',
    '0016809',
    '0011031',
    '0001888',
    '0001535',
    '0002853',
    '0009141',
    '0009677',
    '0016628',
    '0003666',
    '0008214',
    '0012774',
    '0007896',
    '0012890',
    '0011034',
    '0016447',
    '0002242',
    '0013501',
    '0012985',
    '0014770',
    '0018237',
    '0019582',
    '0019767', ]


VALIDATE_INDICES = [
    5, 17, 42, 45, 58, 62, 96, 111, 117, 120, 121, 131, 132,
    152, 160, 248, 263, 264, 291, 293, 295, 299, 316, 320, 336,
    337, 343, 358, 399, 401, 429, 438, 468, 476, 494, 509, 528,
    531, 572, 581, 583, 588, 593, 681, 688, 696, 714, 767, 786,
    810, 825, 836, 841, 883, 917, 937, 942, 970, 974, 980, 1016,
    1043, 1064, 1118, 1121, 1133, 1153, 1155, 1158, 1159, 1173,
    1187, 1219, 1237, 1238, 1259, 1266, 1278, 1296, 1354, 1378,
    1387, 1494, 1508, 1518, 1574, 1601, 1614, 1668, 1673, 1699,
    1712, 1714, 1737, 1841, 1872, 1879, 1901, 1921, 1934, 1961,
    1967, 1978, 2018, 2030, 2039, 2043, 2061, 2113, 2204, 2216,
    2236, 2250, 2274, 2292, 2310, 2342, 2359, 2374, 2382, 2399,
    2415, 2419, 2483, 2502, 2504, 2576, 2589, 2590, 2622, 2624,
    2636, 2651, 2655, 2658, 2659, 2664, 2672, 2706, 2707, 2709,
    2725, 2732, 2761, 2827, 2864, 2866, 2905, 2922, 2929, 2966,
    2972, 2993, 3010, 3025, 3031, 3040, 3041, 3070, 3113, 3124,
    3129, 3137, 3141, 3157, 3183, 3206, 3219, 3247, 3253, 3272,
    3276, 3321, 3328, 3333, 3338, 3341, 3346, 3351, 3396, 3419,
    3430, 3433, 3448, 3455, 3463, 3503, 3526, 3529, 3537, 3555,
    3577, 3584, 3591, 3594, 3597, 3603, 3613, 3615, 3670, 3676,
    3678, 3697, 3723, 3728, 3734, 3745, 3750, 3752, 3779, 3782,
    3813, 3817, 3819, 3854, 3885, 3944, 3947, 3970, 3985, 4011,
    4022, 4071, 4075, 4132, 4158, 4167, 4190, 4194, 4207, 4246,
    4249, 4298, 4307, 4317, 4318, 4319, 4320, 4382, 4399, 4401,
    4407, 4416, 4423, 4484, 4491, 4493, 4517, 4525, 4538, 4578,
    4606, 4609, 4620, 4623, 4637, 4646, 4662, 4668, 4716, 4739,
    4747, 4770, 4774, 4776, 4785, 4800, 4845, 4863, 4891, 4904,
    4922, 4925, 4956, 4963, 4964, 4994, 5011, 5019, 5036, 5038,
    5041, 5055, 5118, 5122, 5130, 5162, 5164, 5178, 5196, 5227,
    5266, 5270, 5273, 5279, 5299, 5310, 5314, 5363, 5375, 5384,
    5393, 5414, 5417, 5433, 5448, 5494, 5505, 5509, 5525, 5566,
    5581, 5602, 5609, 5620, 5653, 5670, 5678, 5690, 5700, 5703,
    5724, 5752, 5765, 5803, 5811, 5860, 5881, 5895, 5912, 5915,
    5940, 5952, 5966, 5977, 5988, 6007, 6037, 6061, 6069, 6080,
    6111, 6127, 6146, 6161, 6166, 6168, 6178, 6182, 6190, 6220,
    6235, 6253, 6270, 6343, 6372, 6379, 6410, 6411, 6442, 6453,
    6481, 6498, 6500, 6509, 6532, 6541, 6543, 6560, 6576, 6580,
    6594, 6595, 6609, 6625, 6629, 6644, 6658, 6673, 6680, 6698,
    6699, 6702, 6705, 6741, 6759, 6785, 6792, 6794, 6809, 6810,
    6830, 6838, 6869, 6871, 6889, 6925, 6995, 7003, 7026, 7029,
    7080, 7082, 7097, 7102, 7116, 7165, 7200, 7232, 7271, 7282,
    7324, 7333, 7335, 7372, 7387, 7407, 7472, 7474, 7482, 7489,
    7499, 7516, 7533, 7536, 7566, 7620, 7654, 7691, 7704, 7722,
    7746, 7750, 7773, 7806, 7821, 7827, 7851, 7873, 7880, 7884,
    7904, 7912, 7948, 7964, 7965, 7984, 7989, 7992, 8035, 8050,
    8074, 8091, 8094, 8113, 8116, 8151, 8159, 8171, 8179, 8194,
    8195, 8239, 8263, 8290, 8295, 8312, 8367, 8374, 8387, 8407,
    8437, 8439, 8518, 8556, 8588, 8597, 8601, 8651, 8657, 8723,
    8759, 8763, 8785, 8802, 8813, 8826, 8854, 8856, 8866, 8918,
    8922, 8923, 8932, 8958, 8967, 9003, 9018, 9078, 9095, 9104,
    9112, 9129, 9147, 9170, 9171, 9197, 9200, 9249, 9253, 9270,
    9282, 9288, 9295, 9321, 9323, 9324, 9347, 9399, 9403, 9417,
    9426, 9427, 9439, 9468, 9486, 9496, 9511, 9516, 9518, 9529,
    9557, 9563, 9564, 9584, 9586, 9591, 9599, 9600, 9601, 9632,
    9654, 9667, 9678, 9696, 9716, 9723, 9740, 9820, 9824, 9825,
    9828, 9863, 9866, 9868, 9889, 9929, 9938, 9953, 9967, 10019,
    10020, 10025, 10059, 10111, 10118, 10125, 10174, 10194,
    10201, 10202, 10220, 10221, 10226, 10242, 10250, 10276,
    10295, 10302, 10305, 10327, 10351, 10360, 10369, 10393,
    10407, 10438, 10455, 10463, 10465, 10470, 10478, 10503,
    10508, 10509, 10809, 11080, 11331, 11607, 11610, 11864,
    12390, 12393, 12396, 12399, 12671, 12921, 12930, 13178,
    13453, 13717, 14499, 14517, 14775, 15297, 15556, 15834,
    15839, 16126, 16127, 16386, 16633, 16644, 16651, 17166,
    17169, 17958, 17959, 17962, 18224, 21176, 21180, 21190,
    21802, 21803, 21806, 22584, 22857, 22858, 22866]


class FlyingChairs2(dataset.Dataset):
    def __init__(self, args, data_root, data_kind='train', aug_params=None):
        super().__init__()
        self.args = args
        self.event_bins = args.event_bins
        self.event_polarity = False if args.no_event_polarity else True
        self.data_root = data_root

        if data_kind[:5] == 'train':
            self.data_split = 'train'
            if len(data_kind) > 5:
                self.data_mode = data_kind[5:]
            else:
                self.data_mode = 'train'
        elif data_kind[:3] == 'val':
            self.data_split = 'val'
            self.data_mode = 'full'
        else:
            raise NotImplementedError(
                "Unsupported data kind {}".format(data_kind))

        self.augmentor = None
        if aug_params is not None:
            self.augmentor = fetch_augmentor(
                is_event=True, is_sparse=False, aug_params=aug_params)

        self.fetch_valids()
        self.data_length = len(self.image1_filenames)

    def fetch_valids(self):

        images_root = os.path.join(self.data_root, self.data_split)
        events_root = os.path.join(self.data_root, "events_" + self.data_split)

        image1_filenames = sorted(
            glob(os.path.join(images_root, "*-img_0.png")))
        image2_filenames = sorted(
            glob(os.path.join(images_root, "*-img_1.png")))
        flow01_filenames = sorted(
            glob(os.path.join(images_root, "*-flow_01.flo")))
        flow10_filenames = sorted(
            glob(os.path.join(images_root, "*-flow_10.flo")))
        event_filenames = sorted(
            glob(os.path.join(events_root, "*-event.hdf5")))

        validate_indices = [
            x for x in VALIDATE_INDICES if x in range(len(image1_filenames))]
        list_of_indices = None
        if self.data_mode[:3] == "val":
            list_of_indices = validate_indices
        elif self.data_mode == "full":
            list_of_indices = range(len(image1_filenames))
        elif self.data_mode == "train":
            list_of_indices = [x for x in range(
                len(image1_filenames)) if x not in validate_indices]
        else:
            raise NotImplementedError(
                "Unsupported data mode {}".format(self.data_mode))

        final_indices = []
        for i in range(len(image1_filenames)):
            im1_base_fileid = (os.path.basename(
                image1_filenames[i])).split("-", 2)[0]
            im2_base_fileid = (os.path.basename(
                image2_filenames[i])).split("-", 2)[0]
            flow_f_base_fileid = (os.path.basename(
                flow01_filenames[i])).split("-", 2)[0]
            flow_b_base_fileid = (os.path.basename(
                flow10_filenames[i])).split("-", 2)[0]
            event_base_fileid = (os.path.basename(
                event_filenames[i])).split("_", 2)[0]

            assert (im1_base_fileid == im2_base_fileid)
            assert (im1_base_fileid == flow_f_base_fileid)
            assert (im1_base_fileid == flow_b_base_fileid)
            assert (im1_base_fileid == event_base_fileid)
            if i in list_of_indices and im1_base_fileid not in FlyingChairs2_BAD_ID:
                final_indices.append(i)

        self.image1_filenames = [image1_filenames[i] for i in final_indices]
        self.image2_filenames = [image2_filenames[i] for i in final_indices]
        self.flow01_filenames = [flow01_filenames[i] for i in final_indices]
        self.flow10_filenames = [flow10_filenames[i] for i in final_indices]
        self.event_filenames = [event_filenames[i] for i in final_indices]

    def load_data_by_index(self, index):
        im1_filename = self.image1_filenames[index]
        im2_filename = self.image2_filenames[index]
        flow01_filename = self.flow01_filenames[index]
        flow10_filename = self.flow10_filenames[index]
        event_filename = self.event_filenames[index]

        im1_nparray = np.array(read_gen(im1_filename)).astype(np.uint8)
        im2_nparray = np.array(read_gen(im2_filename)).astype(np.uint8)
        flow01_nparray = readDenseFlow(flow01_filename)
        flow10_nparray = readDenseFlow(flow10_filename)
        events_nparray = read_event_h5(event_filename)

        return im1_nparray, im2_nparray, flow01_nparray, flow10_nparray, events_nparray

    def __getitem__(self, index):

        index = index % self.data_length

        im1_filename = self.image1_filenames[index]
        basename = os.path.basename(im1_filename)[:6]

        im1_nparray, im2_nparray, flow01_nparray, flow10_nparray, events_nparray = \
            self.load_data_by_index(index)

        height, width = im1_nparray.shape[:2]

        event_voxel_nparray = eventsToVoxel(events_nparray, num_bins=self.event_bins, height=height,
                                            width=width, event_polarity=self.event_polarity, temporal_bilinear=True)

        event_voxel_reversed_nparray = None
        if self.args.isbi:
            event_x = np.flip(events_nparray[:, 0].astype(np.int), axis=0)
            event_y = np.flip(events_nparray[:, 1].astype(np.int), axis=0)
            event_pols = np.flip(-1 * events_nparray[:, 3].astype(np.int), axis=0)
            event_timestamps = events_nparray[:, 2]
            event_timestamps = np.flip(
                event_timestamps.max() - event_timestamps, axis=0)
            events_nparray_bi = np.concatenate(
                (event_x[:, np.newaxis], event_y[:, np.newaxis], event_timestamps[:, np.newaxis], event_pols[:, np.newaxis]), axis=1)
            event_voxel_reversed_nparray = eventsToVoxel(
                events_nparray_bi, num_bins=self.event_bins, height=height, width=width, event_polarity=self.event_polarity, temporal_bilinear=True)
    
        valid = None
        valid10 = None

        if self.augmentor is not None:
            event_voxel_nparray, im1_nparray, im2_nparray, flow01_nparray, flow10_nparray, \
                _, _, event_voxel_reversed_nparray = self.augmentor(
                    event_voxel_nparray, im1_nparray, im2_nparray, flow01_nparray, flow10_nparray, 
                    event_r=event_voxel_reversed_nparray)

        image1 = torch.from_numpy(im1_nparray).permute(2, 0, 1).float()
        image2 = torch.from_numpy(im2_nparray).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow01_nparray).permute(2, 0, 1).float()
        flow10 = torch.from_numpy(flow10_nparray).permute(2, 0, 1).float()

        event_voxel = torch.from_numpy(event_voxel_nparray).float()
        if self.args.isbi:
            reversed_event_voxel = torch.from_numpy(event_voxel_reversed_nparray).float()

        event_valid = torch.norm(event_voxel, p=2, dim=0, keepdim=False) > 0
        event_valid = event_valid.float().unsqueeze(0)

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        valid = valid.float().unsqueeze(0)

        if flow10 is not None:
            if valid10 is not None:
                valid10 = torch.from_numpy(valid)
            else:
                valid10 = (flow10[0].abs() < 1000) & (flow10[1].abs() < 1000)
            valid10 = valid10.float().unsqueeze(0)
        else:
            flow10 = torch.zeros_like(flow, dtype=torch.float32)
            valid10 = torch.zeros_like(valid, dtype=torch.float32)

        if self.args.isbi:
            batch = dict(
                index=index,
                raw_index=index,
                basename=basename,
                height=height,
                width=width,
                image1=image1,
                image2=image2,
                event_voxel=event_voxel,
                event_valid=event_valid,
                reversed_event_voxel=reversed_event_voxel,
                flow_gt=flow,
                flow_valid=valid,
                flow10_gt=flow10,
                flow10_valid=valid10,
            )
        else:
            batch = dict(
                index=index,
                raw_index=index,
                basename=basename,
                height=height,
                width=width,
                image1=image1,
                image2=image2,
                event_voxel=event_voxel,
                event_valid=event_valid,
                flow_gt=flow,
                flow_valid=valid,
            )

        return batch

    def get_raw_events(self, index):
        event_filename = self.event_filenames[index]
        events_nparray = read_event_h5(event_filename)
        return events_nparray

    def get_raw_events_length(self, index):
        return len(self.get_raw_events(index))

    def __len__(self):
        return self.data_length
