import os
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from datasets import *
from utils.misc import *
from utils.transforms import *
from utils.denoise import *
from models.denoise import *
from datasets.mbes import MBESDataset


# Arguments
parser = argparse.ArgumentParser()
## Dataset and loader
parser.add_argument('--raw_data_root', type=str, required=True)
parser.add_argument('--gt_root', type=str, required=True)
parser.add_argument('--noise_min', type=float, default=0.005)
parser.add_argument('--noise_max', type=float, default=0.020)
parser.add_argument('--train_batch_size', type=int, default=32)
# parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--aug_rotate', type=eval, default=True, choices=[True, False])
## Model architecture
parser.add_argument('--supervised', type=eval, default=True, choices=[True, False])
parser.add_argument('--frame_knn', type=int, default=32)
parser.add_argument('--num_train_points', type=int, default=128)
parser.add_argument('--num_clean_nbs', type=int, default=4, help='For supervised training.')
parser.add_argument('--num_selfsup_nbs', type=int, default=8, help='For self-supervised training.')
parser.add_argument('--dsm_sigma', type=float, default=0.01)
parser.add_argument('--score_net_hidden_dim', type=int, default=128)
parser.add_argument('--score_net_num_blocks', type=int, default=4)
## Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=float("inf"))
## Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=1*MILLION)
parser.add_argument('--val_freq', type=int, default=2000)
parser.add_argument('--val_upsample_rate', type=int, default=4)
parser.add_argument('--val_num_visualize', type=int, default=4)
parser.add_argument('--val_noise', type=float, default=0.015)
parser.add_argument('--ld_step_size', type=float, default=0.2)
parser.add_argument('--tag', type=str, default=None)
args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root,
                              prefix='MBES_patches',
                              postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    log_hyperparams(writer, log_dir, args)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
logger.info('Loading datasets')

train_dset = MBESDataset(
    raw_data_root=args.raw_data_root,
    gt_root=args.gt_root,
    split='tmp',
    transform=Compose([NormalizeZ(),
                    #    AddNoiseToZ(noise_std_max=args.noise_max,
                    #                noise_std_min=args.noise_min),
    # transform=standard_train_transforms(noise_std_max=args.noise_max,
    #                                     noise_std_min=args.noise_min,
    #                                     rotate=args.aug_rotate)
    ])
)
print(f"train_dset: {len(train_dset)}")
val_dset = MBESDataset(
    raw_data_root=args.raw_data_root,
    gt_root=args.gt_root,
    split='tmp-val',
    transform=Compose([NormalizeZ()]),
    # transform=standard_train_transforms(noise_std_max=args.noise_max,
    #                                     noise_std_min=args.noise_min,
    #                                     rotate=args.aug_rotate)
)
print(f"val_dset: {len(val_dset)}")
train_iter = get_data_iterator(DataLoader(train_dset, batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=True))

# Model
logger.info('Building model...')
model = DenoiseNet(args).to(args.device)
logger.info(repr(model))

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
)

# Train, validate and test
def train(it):
    # Load data
    batch = next(train_iter)
    print(f"pcl noisy orig shape: {batch['pcl_noisy'].shape}")
    pcl_noisy = batch['pcl_noisy'].reshape(args.train_batch_size, -1, 3).to(args.device)
    pcl_clean = batch['pcl_clean'].reshape(args.train_batch_size, -1, 3).to(args.device)

    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    if args.supervised:
        loss = model.get_supervised_loss(pcl_noisy=pcl_noisy, pcl_clean=pcl_clean)
    else:
        loss = model.get_selfsupervised_loss(pcl_noisy=pcl_noisy)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()

    # Logging
    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.6f' % (
        it, loss.item(), orig_grad_norm,
    ))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush() 

def validate(it):
    cd = []
    for i, data in enumerate(tqdm(val_dset, desc='Validate')):
        pcl_noisy = data['pcl_noisy'].reshape(-1, 3).to(args.device)
        pcl_clean = data['pcl_clean'].reshape(-1, 3).to(args.device)
        pcl_denoised = patch_based_denoise(model, pcl_noisy, ld_step_size=args.ld_step_size)

        cd.append(pytorch3d.loss.chamfer_distance(pcl_denoised.unsqueeze(0),
                                                  pcl_clean.unsqueeze(0),
                                                  batch_reduction='mean')[0].item())

    avg_chamfer = np.mean(cd)

    logger.info('[Val] Iter %04d | CD %.6f  ' % (it, avg_chamfer))
    writer.add_scalar('val/chamfer', avg_chamfer, it)
    writer.flush()

    # scheduler.step(avg_chamfer)
    return avg_chamfer

# Main loop
logger.info('Start training...')
try:
    validate(0)
    for it in range(1, args.max_iters+1):
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            cd_loss = validate(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict(),
            }
            ckpt_mgr.save(model, args, cd_loss, opt_states, step=it)
            # ckpt_mgr.save(model, args, 0, opt_states, step=it)

except KeyboardInterrupt:
    logger.info('Terminating...')
