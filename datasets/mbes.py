import os
import torch
import numpy as np
from tqdm.auto import tqdm
import re
from torch.utils.data import Dataset


class MBESDataset(Dataset):

    def __init__(self, raw_data_root, gt_root, split, transform=None, use_ping_idx=False):
        self.raw_data_root = os.path.join(raw_data_root, split)
        self.gt_root = os.path.join(gt_root, split)

        self.raw_data_files = sorted([os.path.join(self.raw_data_root, f) for f in os.listdir(self.raw_data_root)
                                      if f.split('.')[-1] == 'npz'])
        self.gt_data_files = sorted([os.path.join(self.gt_root, f) for f in os.listdir(self.gt_root)
                                     if f.split('.')[-1] == 'npz'])
        print(f'len raw_data_files: {len(self.raw_data_files)}')
        print(f'len gt_data_files: {len(self.gt_data_files)}')
        assert len(self.raw_data_files) == len(self.gt_data_files)

        self.transform = transform
        self.use_ping_idx = use_ping_idx

    def _use_ping_beam_range_representation(self, pcl, angles):
        num_pings, num_beams = pcl.shape[:2]
        ping_idx = np.arange(num_pings).reshape(-1, 1).repeat(num_beams, axis=1)
        beam_idx = np.arange(num_beams).reshape(1, -1).repeat(num_pings, axis=0)
        z_range = pcl[:, :, 2] / np.cos(np.radians(angles))
        pcl[:, :, 0] = ping_idx
        pcl[:, :, 1] = beam_idx
        pcl[:, :, 2] = z_range
        return pcl

    def _load_raw_data(self, idx):
        data = np.load(self.raw_data_files[idx], allow_pickle=True)
        angles = data["angle"]
        pcl = np.stack([data["X"], data["Y"], data["Z_relative"]], axis=-1)
        pcl = pcl.reshape(-1, 3)

        # Load the rejection mask and record the indices of rejected points, indexed as 1D array
        rejected = np.argwhere(data["rejected"].flatten(), ).flatten()
        return pcl, rejected, angles

    def _load_gt_data(self, idx):
        data = np.load(self.gt_data_files[idx], allow_pickle=True)
        # filter out invalid draping points
        pcl = data['data']
        valid_mask = data['valid_mask']
        pcl = pcl[valid_mask]
        return pcl, valid_mask

    def __len__(self):
        return len(self.raw_data_files)

    def __getitem__(self, idx):
        pcl_raw, rejected, angles = self._load_raw_data(idx)
        pcl_gt, valid_mask = self._load_gt_data(idx)
        if self.use_ping_idx:
            pcl_raw = self._use_ping_beam_range_representation(pcl_raw, angles)
            pcl_gt = self._use_ping_beam_range_representation(pcl_gt, angles)

        data = {
            'pcl_clean': torch.from_numpy(pcl_gt).clone().float(),
            'pcl_raw': torch.from_numpy(pcl_raw).clone().float(),
            'name': idx,
            'valid_mask': torch.from_numpy(valid_mask).clone().bool(),
            'rejected': torch.from_numpy(rejected).clone().long(),
        }

        if self.transform is not None:
            data = self.transform(data)
        return data