import os
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from datasets import *
from utils.misc import *
from utils.transforms import *
from utils.denoise import *
from models.denoise import *
from datasets.mbes import MBESDataset
from models.utils import compute_mbes_denoising_metrics, compute_mbes_outlier_rejection_metrics
from collections import defaultdict

import wandb


# Arguments
parser = argparse.ArgumentParser()
## Dataset and loader
parser.add_argument('--raw_data_root', type=str, required=True)
parser.add_argument('--gt_root', type=str, required=True)
parser.add_argument('--noise_min', type=float, default=0.005)
parser.add_argument('--noise_max', type=float, default=0.020)
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--global_norm_z', action='store_true',
                    help='If True, normalize Z globally using pre-computed max z per data split.')
# parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--aug_rotate', type=eval, default=True, choices=[True, False])
## Model architecture
parser.add_argument('--supervised', type=eval, default=True, choices=[True, False])
parser.add_argument('--frame_knn', type=int, default=32)
parser.add_argument('--num_train_points', type=int, default=2048)
parser.add_argument('--num_clean_nbs', type=int, default=32, help='For supervised training.')
parser.add_argument('--num_selfsup_nbs', type=int, default=32, help='For self-supervised training.')
parser.add_argument('--dsm_sigma', type=float, default=0.01)
parser.add_argument('--score_net_hidden_dim', type=int, default=32)
parser.add_argument('--score_net_num_blocks', type=int, default=3)
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
parser.add_argument('--ld_step_size', type=float, default=0.2)
parser.add_argument('--tag', type=str, default=None)
## Denoising
parser.add_argument('--denoise_knn', type=int, default=32)
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('--test', action='store_true', help='Test mode')
args = parser.parse_args()
seed_all(args.seed)

logger_name = 'train' if not args.test else 'test'
# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root,
                              prefix='MBES_patches',
                              postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger(logger_name, log_dir)
    wandb.init(config=args,
               project='MBES-denoising-3d',
               name=args.tag)
    wandb.tensorboard.patch(root_logdir=log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    log_hyperparams(writer, log_dir, args)
else:
    logger = get_logger(logger_name, None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
logger.info('Loading datasets')

global_norm_z = {
    'train': 45.53,
    'validation': 31.44,
    'test': 28.82,
}
instance_norm_z = {
    'train': None,
    'validation': None,
    'test': None,
}
if args.global_norm_z:
    norm_z_dict = global_norm_z
else:
    norm_z_dict = instance_norm_z

def load_training_data():
    train_dset = MBESDataset(
        use_ping_idx=False,
        raw_data_root=args.raw_data_root,
        gt_root=args.gt_root,
        split='train_data_merged',
        transform=Compose([
            NormalizeZ(z_scale=norm_z_dict['train']),
            RandomRotate(axis=2),
        ])
    )

    print(f"train_dset: {len(train_dset)}")
    val_dset = MBESDataset(
        use_ping_idx=False,
        raw_data_root=args.raw_data_root,
        gt_root=args.gt_root,
        split='validation',
        transform=Compose([
            NormalizeZ(z_scale=norm_z_dict['validation']),
            RandomRotate(axis=2),
        ])
    )
    print(f"val_dset: {len(val_dset)}")
    return train_dset, val_dset

def load_test_data():
    test_dset = MBESDataset(
        use_ping_idx=False,
        raw_data_root=args.raw_data_root,
        gt_root=args.gt_root,
        split='test_data.npz',
        transform=Compose([
            NormalizeZ(z_scale=norm_z_dict['test']),
        ])
    )
    return test_dset

train_dset, val_dset = load_training_data()
train_iter = get_data_iterator(DataLoader(train_dset, 
                                          batch_size=1,
                                          num_workers=args.num_workers,
                                          shuffle=True))
test_dset = load_test_data()

# Model
logger.info('Building model...')
model = DenoiseNet(args).to(args.device)
if args.ckpt_path is not None:
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt['state_dict'])
logger.info(repr(model))

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Train, validate and test
def train(it):
    model.train()
    # Manual batch construction
    batch_size = args.train_batch_size
    # Load data
    for _ in range(batch_size):
        batch = next(train_iter)
        # print(f"pcl noisy orig shape: {batch['pcl_noisy'].shape}")
        pcl_noisy = batch['pcl_noisy'].reshape(1, -1, 3).to(args.device)
        pcl_clean = batch['pcl_clean'].reshape(1, -1, 3).to(args.device)
        rejected = batch['rejected'].reshape(1, -1).to(args.device)

        # Forward
        if args.supervised:
            loss = model.get_supervised_loss(pcl_noisy=pcl_noisy, pcl_clean=pcl_clean,
                                            rejected=rejected)
        else:
            loss = model.get_selfsupervised_loss(pcl_noisy=pcl_noisy)

        # Backward
        loss.backward()

    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()


    # Logging
    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.6f' % (
        it, loss.item(), orig_grad_norm,
    ))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush() 

def validate(it, dset, store_results=False, ckpt_path=None):
    denoising_metrics_denorm = defaultdict(list)
    binary_metrics = defaultdict(list)

    if store_results:
        if ckpt_path is None:
            raise ValueError('ckpt_path must be provided if store_results is True')
        if args.global_norm_z:
            norm_str = 'global'
        else:
            norm_str = 'instance'
        save_dir = f'{ckpt_path}_results_{norm_str}_norm'
        os.makedirs(save_dir, exist_ok=False)
        

    for i, data in enumerate(tqdm(dset, desc='Validate')):
        pcl_noisy = data['pcl_noisy'].reshape(-1, 3).to(args.device)
        pcl_clean = data['pcl_clean'].reshape(-1, 3).to(args.device)
        gt_rejected_mask = data['rejected'].to(args.device) # Shape: (N,)
        valid_mask = data['valid_mask'].to(args.device)
        scale_xy = data['scale_xy'].to(args.device)
        if args.global_norm_z:
            scale_z = data['scale_z']
        else:
            scale_z = data['scale_z'].to(args.device)
        center = data['center'].to(args.device)

        pcl_denoised, init_grad = patch_based_denoise(model, pcl_noisy, ld_step_size=args.ld_step_size,
                                                      denoise_knn=args.denoise_knn)
        denoising_results = compute_mbes_denoising_metrics(pcl_denoised, pcl_clean, valid_mask=valid_mask, denormalize=True,
                                              scale_xy=scale_xy, scale_z=scale_z, center=center)
        outlier_results = compute_mbes_outlier_rejection_metrics(init_grad.flatten(), gt_rejected_mask.flatten(),
                                                                thresh=5.)

        for k, v in denoising_results.items():
            denoising_metrics_denorm[k].append(v)
        for k, v in outlier_results.items():
            binary_metrics[k].append(v)

        if store_results:
            idx = data['name']
            pcl_denoised = pcl_denoised.cpu().numpy()
            pcl_clean = pcl_clean.cpu().numpy()
            pcl_noisy = pcl_noisy.cpu().numpy()
            init_grad = init_grad.cpu().numpy()
            gt_rejected_mask = gt_rejected_mask.cpu().numpy()
            valid_mask = valid_mask.cpu().numpy()
            scale_xy = scale_xy.cpu().numpy()
            if args.global_norm_z:
                scale_z = scale_z
            else:
                scale_z = scale_z.cpu().numpy()
            center = center.cpu().numpy()
            np.savez(f'{save_dir}/patch_based_denoise_{i}_patchid_{idx}.npz', pcl_denoised=pcl_denoised, pcl_clean=pcl_clean,
                     pcl_noisy=pcl_noisy, init_grad=init_grad, gt_rejected_mask=gt_rejected_mask, valid_mask=valid_mask,
                     scale_xy=scale_xy, scale_z=scale_z, center=center)

    avg_metrics = {f'{k}_denorm': np.mean(v) for k, v in denoising_metrics_denorm.items()}
    avg_metrics.update({
        f'{k}_sum': np.sum(v) for k, v in binary_metrics.items()
    })
    
    # compute accuracy, precision, recall, F1-score from binary_metrics using the _sum values
    TP_sum = avg_metrics['TP_sum']
    FP_sum = avg_metrics['FP_sum']
    TN_sum = avg_metrics['TN_sum']
    FN_sum = avg_metrics['FN_sum']
    avg_metrics['accuracy'] = (TP_sum + TN_sum) / (TP_sum + FP_sum + TN_sum + FN_sum)
    avg_metrics['precision'] = TP_sum / (TP_sum + FP_sum)
    avg_metrics['recall'] = TP_sum / (TP_sum + FN_sum)
    # compute F1-score
    avg_metrics['f1_score'] = 2 * (avg_metrics['precision'] * avg_metrics['recall']) / (avg_metrics['precision'] + avg_metrics['recall'])

    # logger.info('[Val] Iter %04d | CD %.6f  ' % (it, avg_chamfer))
    # writer.add_scalar('val/chamfer', avg_chamfer, it)
    logger.info(f'[Val] Iter {it:4d} Denoise | CD {avg_metrics["cd_denorm"]:.6f} | |z| {avg_metrics["z_abs_diff_denorm"]:.6f} | z_rmse {avg_metrics["z_rmse_denorm"]:.6f}')
    # log accuracy, precision, recall, F1-score
    logger.info(f'[Val] Iter {it:4d} Outlier | Accuracy {avg_metrics["accuracy"]*100:.2f}% | Precision {avg_metrics["precision"]*100:.2f}% | Recall {avg_metrics["recall"]*100:.2f}% | F1-score {avg_metrics["f1_score"]:.6f}')
    logger.info(f'[Val] Iter {it:4d} Outlier | TP {TP_sum} | FP {FP_sum} | TN {TN_sum} | FN {FN_sum}')
    for k, v in avg_metrics.items():
        writer.add_scalar(f'val/{k}', v, it)
    writer.flush()

    # scheduler.step(avg_chamfer)
    # return avg_chamfer
    return avg_metrics['z_abs_diff_denorm']

if args.test:
    logger.info('Testing...')
    validate(0, test_dset, store_results=True, ckpt_path=args.ckpt_path)
    exit()

# Main loop
logger.info('Start training...')
try:
    validate(0, val_dset)
    for it in range(1, args.max_iters+1):
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            cd_loss = validate(it, val_dset)
            scheduler.step(cd_loss)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            ckpt_mgr.save(model, args, cd_loss, opt_states, step=it)
            # ckpt_mgr.save(model, args, 0, opt_states, step=it)

except KeyboardInterrupt:
    logger.info('Terminating...')
    wandb.finish()
