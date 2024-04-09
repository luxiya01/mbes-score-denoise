import os
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence

from datasets import *
from datasets.mbes_pcl import MBESPatchDataset
from utils.misc import *
from utils.transforms import *
from utils.denoise import *
from models.denoise_mbes import *
from models.utils import chamfer_distance_unit_sphere


# Arguments
parser = argparse.ArgumentParser()
## Dataset and loader
parser.add_argument('--train_batch_size', type=int, default=32)
# parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
## Model architecture
#parser.add_argument('--frame_knn', type=int, default=32)
parser.add_argument('--num_train_points', type=int, default=128)
parser.add_argument('--dsm_sigma', type=float, default=10)
parser.add_argument('--score_net_hidden_dim', type=int, default=32)
parser.add_argument('--score_net_num_blocks', type=int, default=2)
## Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=float("inf"))
## Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=10000)#1*MILLION)
parser.add_argument('--val_freq', type=int, default=200)
parser.add_argument('--val_num_visualize', type=int, default=4)
parser.add_argument('--ld_step_size', type=float, default=0.2)
parser.add_argument('--tag', type=str, default=None)
args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='MBES_', postfix='_' + args.tag if args.tag is not None else '')
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
train_dset = MBESPatchDataset(
    data_path='/home/li/mbes-cleaning/data-dotson-east-unfiltered/merged/all_data.npz',
    gt_path='/home/li/mbes-cleaning/data-dotson-east-unfiltered/merged/all_data_draping_2.36mesh_gt_5.0m.npy',
    transform=None,
    pings_subset=(0, 2000),
    pings_per_patch=2,
)

val_dset = MBESPatchDataset(
    data_path='/home/li/mbes-cleaning/data-dotson-east-unfiltered/merged/all_data.npz',
    gt_path='/home/li/mbes-cleaning/data-dotson-east-unfiltered/merged/all_data_draping_2.36mesh_gt_5.0m.npy',
    transform=None,
    pings_subset=(1000, 1200),
    pings_per_patch=1,
)

def custom_collate(data):
    pcl_clean = [d['pcl_clean'].clone().detach() for d in data]
    pcl_noisy = [d['pcl_noisy'].clone().detach() for d in data]
    pcl_noisy_mean = [d['pcl_noisy_mean'].clone().detach() for d in data]
    pcl_length = [pcl.shape[0] for pcl in pcl_clean]

    pcl_clean = pad_sequence(pcl_clean, batch_first=True)
    pcl_noisy = pad_sequence(pcl_noisy, batch_first=True)
    return {
        'pcl_clean': pcl_clean,
        'pcl_noisy': pcl_noisy,
        'pcl_noisy_mean': torch.stack(pcl_noisy_mean, axis=0),
        'pcl_length': torch.LongTensor(pcl_length),
    }

train_iter = get_data_iterator(
    DataLoader(train_dset,
               batch_size=args.train_batch_size,
               num_workers=args.num_workers,
               shuffle=True,
               collate_fn=custom_collate))

# Model
logger.info('Building model...')
model = DenoiseNet(args).to(args.device)
logger.info(repr(model))

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True,
)

# Train, validate and test
def train(it):
    # Load data
    batch = next(train_iter)
    pcl_noisy = batch['pcl_noisy'].to(args.device)
    pcl_clean = batch['pcl_clean'].to(args.device)
    pcl_length = batch['pcl_length'].to(args.device)

    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    loss = model.get_supervised_loss(pcl_noisy=pcl_noisy, pcl_clean=pcl_clean,
                                        pcl_length=pcl_length)

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
    all_clean = []
    all_noisy = []
    all_denoised = []
    for i, data in enumerate(tqdm(val_dset, desc='Validate')):
        pcl_noisy = data['pcl_noisy'].to(args.device)
        pcl_clean = data['pcl_clean'].to(args.device)
        all_noisy.append(pcl_noisy)
        all_clean.append(pcl_clean)

        pcl_denoised = mbes_denoise(model, pcl_noisy, ld_step_size=args.ld_step_size)
        all_denoised.append(pcl_denoised)
    all_clean = torch.cat(all_clean, dim=0)
    all_denoised = torch.cat(all_denoised, dim=0)
    all_noisy = torch.cat(all_noisy, dim=0)

    #writer.add_mesh('val/pcl', all_denoised[:args.val_num_visualize], global_step=it)
    #writer.add_mesh('gt/pcl', all_clean[:args.val_num_visualize], global_step=it)
    np.save('val_pcl.npy', all_denoised.cpu().numpy())
    np.save('gt_pcl.npy', all_clean.cpu().numpy())
    np.save('noisy_pcl.npy', all_noisy.cpu().numpy())

    writer.add_mesh('val/pcl', all_denoised.reshape(1, -1, 3), global_step=it)
    writer.add_mesh('gt/pcl', all_clean.reshape(1, -1, 3), global_step=it)


    avg_chamfer = pytorch3d.loss.chamfer_distance(
        all_denoised.unsqueeze(0),
        all_clean.unsqueeze(0),
        batch_reduction='mean',
        point_reduction='mean')[0].item()
    avg_diff = torch.norm(all_denoised - all_clean, dim=-1).mean().item()

    logger.info('[Val] Iter %04d | CD %.6f  | Diff %.6f' % (it, avg_chamfer, avg_diff))
    writer.add_scalar('val/chamfer', avg_chamfer, it)
    writer.add_scalar('val/diff', avg_diff, it)

    writer.flush()

    # scheduler.step(avg_chamfer)
    return avg_chamfer, avg_diff

# Main loop
logger.info('Start training...')
try:
    for it in range(1, args.max_iters+1):
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            cd_loss, diff_loss = validate(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                # 'scheduler': scheduler.state_dict(),
            }
            ckpt_mgr.save(model, args, diff_loss, opt_states, step=it)
            # ckpt_mgr.save(model, args, 0, opt_states, step=it)
            scheduler.step(diff_loss)

except KeyboardInterrupt:
    logger.info('Terminating...')
