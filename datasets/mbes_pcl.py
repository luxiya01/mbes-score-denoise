import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class MBESPatchDataset(Dataset):

    def __init__(self, data_path, gt_path, transform, pings_subset=(0, 5000),
                 subset='train', pings_per_patch=32, normalize_xy=False):
        super().__init__()
        self.pings_subset = pings_subset
        self.pings_per_patch = pings_per_patch
        self.transform = transform
        self.subset = subset
        self.normalize_xy = normalize_xy
        self.raw_cloud, self.raw_rejection_mask = self._load_raw_pointcloud(data_path)
        self.gt_cloud = self._load_gt_pointcloud(gt_path)

    def _normalize_xy(self, pcl):
        #TODO: normalize xy coordinates using mean and std OR [0, 1] OR [-1, 1]?
        pcl[:, 0] = (pcl[:, 0] - np.mean(pcl[:, 0])) / np.std(pcl[:, 0])
        pcl[:, 1] = (pcl[:, 1] - np.mean(pcl[:, 1])) / np.std(pcl[:, 1])
        return pcl

    def _demean_points(self, pcl):
        pcl -= np.mean(pcl, axis=0)
        return pcl

    def _load_raw_pointcloud(self, data_path):
        data = np.load(data_path, allow_pickle=True)
        # angles = data['angle'][:, ::-1] # reverse the order of the angles (NED to ENU)
        pcl = np.stack([
            data['X'], data['Y'], data['Z_relative']#, angles
        ], axis=-1)
        print(f'pcl.shape: {pcl.shape}')
        pcl = pcl[self.pings_subset[0]:self.pings_subset[1], ...]
        print(f'pcl.shape: {pcl.shape}')

        rejection_mask = data['rejected'][self.pings_subset[0]:self.pings_subset[1], :]
        return pcl, rejection_mask

    def _load_gt_pointcloud(self, gt_path):
        gt = np.load(gt_path)[self.pings_subset[0]:self.pings_subset[1], :, :].astype(np.float32)
        print(f'gt.shape: {gt.shape}')
        return gt


    def __len__(self):
        return self.raw_cloud.shape[0] // self.pings_per_patch

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError('Index out of range.')
        pcl_clean = self.gt_cloud[idx * self.pings_per_patch:(idx + 1) * self.pings_per_patch].reshape(self.pings_per_patch, -1, 3)
        pcl_noisy = self.raw_cloud[idx * self.pings_per_patch:(idx + 1) * self.pings_per_patch].reshape(self.pings_per_patch, -1, 3)

        # filter out pings without ground truth
        if self.subset == 'train':
            gt_mask = np.where(pcl_clean[:, :, -1] != 0)
            pcl_clean = pcl_clean[gt_mask]
            pcl_noisy = pcl_noisy[gt_mask]

        if self.normalize_xy:
            pcl_noisy = self._normalize_xy(pcl_noisy)
            pcl_clean = self._normalize_xy(pcl_clean)
        else:
            # compute mean XYZ of the noisy point cloud
            pcl_noisy_mean = np.mean(pcl_noisy.reshape(-1, 3), axis=0)
            pcl_noisy -= pcl_noisy_mean
            pcl_clean -= pcl_noisy_mean

        data = {
            'pcl_clean': pcl_clean,
            'pcl_noisy': pcl_noisy,
        }

        if self.transform is not None:
            data = self.transform(data)
        
        data['pcl_noisy'] = torch.from_numpy(data['pcl_noisy']).to(torch.float32).reshape(-1, 3)
        data['pcl_clean'] = torch.from_numpy(data['pcl_clean']).to(torch.float32).reshape(-1, 3)

        return data