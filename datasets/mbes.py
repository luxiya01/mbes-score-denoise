import os
import torch
import numpy as np
from tqdm.auto import tqdm
import re
from torch.utils.data import Dataset


class MBESDataset(Dataset):

    def __init__(self, raw_data_root, gt_root, split, transform=None):
        self.raw_data_root = os.path.join(raw_data_root, split)
        self.gt_root = os.path.join(gt_root, split)

        self.raw_data_files = sorted([os.path.join(self.raw_data_root, f) for f in os.listdir(self.raw_data_root)])
        self.gt_data_files = sorted([os.path.join(self.gt_root, f) for f in os.listdir(self.gt_root)])
        assert len(self.raw_data_files) == len(self.gt_data_files)

        self.transform = transform

    def _load_raw_data(self, idx):
        data = np.load(self.raw_data_files[idx], allow_pickle=True)
        pcl = np.stack([data["X"], data["Y"], data["Z_relative"]], axis=-1).reshape(-1, 3)
        return pcl

    def _load_gt_data(self, idx):
        data = np.load(self.gt_data_files[idx], allow_pickle=True)
        # filter out invalid draping points
        pcl = data['data']
        valid_mask = data['valid_mask']
        pcl = pcl[valid_mask]
        return pcl

    def __len__(self):
        return len(self.raw_data_files)

    def __getitem__(self, idx):
        pcl_raw = self._load_raw_data(idx)
        pcl_gt = self._load_gt_data(idx)
        data = {
            'pcl_clean': torch.from_numpy(pcl_gt).clone().float(),
            'pcl_raw': torch.from_numpy(pcl_raw).clone().float(),
            'name': idx,
        }

        if self.transform is not None:
            data = self.transform(data)
        return data