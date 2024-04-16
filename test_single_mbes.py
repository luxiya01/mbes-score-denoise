import os
import time
import argparse
import torch
from tqdm.auto import tqdm

from utils.misc import *
from utils.denoise import *
from utils.transforms import *
from models.denoise import *

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./pretrained/ckpt.pt')
parser.add_argument('--input_data', type=str, default='./data/mbes-data/patch_1.npz')
parser.add_argument('--output_data', type=str, default='./data/mbes-data/patch_1_denoised.npz')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=2020)
args = parser.parse_args()
seed_all(args.seed)

# Model
ckpt = torch.load(args.ckpt, map_location=args.device)
model = DenoiseNet(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])


# Point cloud
data = np.load(args.input_data)
pcl = np.stack([data['X'], data['Y'], data['Z']], axis=-1).reshape(-1, 3)
pcl = torch.FloatTensor(pcl)
pcl, center, scale = NormalizeUnitSphere.normalize(pcl)
pcl = pcl.to(args.device)

print('[INFO] Start denoising...')
pcl_denoised = patch_based_denoise(model, pcl).cpu()
pcl_denoised = pcl_denoised * scale + center
print('[INFO] Finished denoising.')

print('[INFO] Saving denoised point cloud to: %s' % args.output_data)
np.savetxt(args.output_data, pcl_denoised, fmt='%.8f')
