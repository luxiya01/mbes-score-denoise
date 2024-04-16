import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import re

class MBESDataset(Dataset):

    def __init__(self, root, split, transform=None):
        super().__init__()
        self.pcl_dir = os.path.join(root, split)
        self.transform = transform
        self.pointclouds = []
        self.pointcloud_names = []
        self._load_pointclouds()

    def _load_pointclouds(self):
        for fn in tqdm(os.listdir(self.pcl_dir), desc='Loading'):
            if not re.match(r'patch_\d+\.npz', fn):
                continue
            pcl_path = os.path.join(self.pcl_dir, fn)
            data = np.load(pcl_path)
            # load the noisy raw data
            pcl = np.stack([data['X'], data['Y'], data['Z']], axis=-1).reshape(-1, 3)
            pcl = torch.FloatTensor(pcl)
            self.pointclouds.append(pcl)
            self.pointcloud_names.append(fn.split('.')[0])

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {
            # TODO: note that the raw data is actually noisy...
            'pcl_clean': self.pointclouds[idx].clone(),
            'name': self.pointcloud_names[idx]
        }
        if self.transform is not None:
            data = self.transform(data)
        return data