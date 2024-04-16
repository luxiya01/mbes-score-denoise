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
import wandb

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/example.yaml")
# The following arguments are for the wandb sweep only
parser.add_argument("--lr", type=float, required=False)
parser.add_argument("--dsm_sigma", type=float, required=False)
parser.add_argument("--weight_decay", type=float, required=False)
parser.add_argument("--lr_decay", type=float, required=False)

main_args = parser.parse_args()
args = load_config(main_args.config)
if hasattr(main_args, "lr"):
    args.lr = main_args.lr
if hasattr(main_args, "dsm_sigma"):
    args.dsm_sigma = main_args.dsm_sigma
if hasattr(main_args, "weight_decay"):
    args.weight_decay = main_args.weight_decay
if hasattr(main_args, "lr_decay"):
    args.lr_decay = main_args.lr_decay
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(
        args.log_root,
        prefix="MBES_",
        postfix="_" + args.tag if args.tag is not None else "",
    )
    logger = get_logger("train", log_dir)
    # N.B. Wandb sync only works if wandb.init() is called before the writer is created!!!
    wandb.init(
        config=args,
        project="MBES-denoising",
        name=args.tag,
    )
    wandb.tensorboard.patch(root_logdir=log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    log_hyperparams(writer, log_dir, args)
else:
    logger = get_logger("train", None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)
logger.info("Logging to %s" % log_dir)


def get_data_transform(args_dataset):
    transforms_dict = {
        "add_noise_to_clean": AddNoise,
        "add_noise_to_noisy": AddNoiseToNoisyPCL,
        "rotate": RandomRotate,
    }

    if args_dataset.transform is None:
        transforms = None
    else:
        transforms = []
        if "noise" in args_dataset.transform:
            transforms.append(
                transforms_dict[args_dataset.transform.noise.type](
                    noise_std_min=args_dataset.transform.noise.min,
                    noise_std_max=args_dataset.transform.noise.max,
                )
            )
        if "rotate" in args_dataset.transform:
            transforms.append(
                transforms_dict["rotate"](
                    axis=args_dataset.transform.rotate.axis,
                    degrees=args_dataset.transform.rotate.max_angle,
                )
            )
    return Compose(transforms)


# Datasets and loaders
logger.info("Loading datasets")
train_dset = MBESPatchDataset(
    data_path=args.train_dataset.data_path,
    gt_path=args.train_dataset.gt_path,
    transform=get_data_transform(args.train_dataset),
    pings_subset=args.train_dataset.pings_subset,
    pings_per_patch=args.train_dataset.pings_per_patch,
)

val_dset = MBESPatchDataset(
    data_path=args.val_dataset.data_path,
    gt_path=args.val_dataset.gt_path,
    transform=get_data_transform(args.val_dataset),
    pings_subset=args.val_dataset.pings_subset,
    pings_per_patch=args.val_dataset.pings_per_patch,
)


def custom_collate(data):
    pcl_clean = [d["pcl_clean"].clone().detach() for d in data]
    pcl_noisy = [d["pcl_noisy"].clone().detach() for d in data]
    pcl_noisy_mean = [d["pcl_noisy_mean"].clone().detach() for d in data]
    pcl_length = [pcl.shape[0] for pcl in pcl_clean]

    pcl_clean = pad_sequence(pcl_clean, batch_first=True)
    pcl_noisy = pad_sequence(pcl_noisy, batch_first=True)
    return {
        "pcl_clean": pcl_clean,
        "pcl_noisy": pcl_noisy,
        "pcl_noisy_mean": torch.stack(pcl_noisy_mean, axis=0),
        "pcl_length": torch.LongTensor(pcl_length),
    }


train_loader = DataLoader(
    train_dset,
    batch_size=args.train_batch_size,
    num_workers=args.num_workers,
    shuffle=True,
    collate_fn=custom_collate,
)
val_loader = DataLoader(
    val_dset,
    batch_size=args.val_batch_size,
    num_workers=args.num_workers,
    shuffle=False,
    collate_fn=custom_collate,
)

# Model
logger.info("Building model...")
model = DenoiseNet(args).to(args.device)
logger.info(repr(model))

# Optimizer and scheduler
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)


# Train, validate and test
def train_for_one_epoch(epoch_num):
    model.train()
    running_loss = 0.0
    running_grad = 0.0
    len_data_loader = len(train_loader)
    for i, data in enumerate(train_loader):
        pcl_noisy = data["pcl_noisy"].to(args.device)
        pcl_clean = data["pcl_clean"].to(args.device)
        pcl_length = data["pcl_length"].to(args.device)

        # Reset grad
        optimizer.zero_grad()

        # Forward
        loss = model.get_supervised_loss(
            pcl_noisy=pcl_noisy, pcl_clean=pcl_clean, pcl_length=pcl_length
        )

        # Backward and optimize
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # Adjust learning weights
        optimizer.step()

        # Logging
        running_loss += loss.item()
        running_grad += orig_grad_norm
        if i % args.log_interval == 0:

            avg_loss_per_iter = running_loss / args.log_interval
            avg_grad_per_iter = running_grad / args.log_interval
            logger.info(
                "[Train] Iter %04d | Loss %.6f | Grad %.6f"
                % (
                    i,
                    avg_loss_per_iter,
                    avg_grad_per_iter,
                )
            )

            step = epoch_num * len_data_loader + i
            writer.add_scalar("train/loss", avg_loss_per_iter, step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
            writer.add_scalar("train/grad_norm", avg_grad_per_iter, step)
            writer.flush()
            running_loss = 0.0
            running_grad = 0.0


@torch.no_grad()
def validate(it):
    model.eval()
    all_clean = []
    all_noisy = []
    all_denoised = []
    all_length = []
    all_chamfer = []
    all_point_corr_distance = []
    all_loss = []
    for i, data in enumerate(tqdm(val_loader, desc="Validate")):
        pcl_noisy = data["pcl_noisy"].squeeze(dim=0).to(args.device)
        pcl_clean = data["pcl_clean"].squeeze(dim=0).to(args.device)
        pcl_length = data["pcl_length"].squeeze(dim=0).to(args.device)
        all_noisy.append(pcl_noisy)
        all_clean.append(pcl_clean)
        all_length.append(pcl_length)

        loss = model.get_supervised_loss(
            pcl_noisy.unsqueeze(0), pcl_clean.unsqueeze(0), pcl_length.unsqueeze(0)
        )
        all_loss.append(loss.item())

        pcl_denoised = mbes_denoise(model, pcl_noisy, ld_step_size=args.ld_step_size)
        all_denoised.append(pcl_denoised)

        chamfer = pytorch3d.loss.chamfer_distance(
            pcl_denoised.unsqueeze(0),
            pcl_clean.unsqueeze(0),
            batch_reduction="mean",
            point_reduction="mean",
        )[0].item()
        all_chamfer.append(chamfer)
        point_corr_distance = (
            torch.linalg.norm(pcl_denoised - pcl_clean, dim=-1).mean().item()
        )
        all_point_corr_distance.append(point_corr_distance)
    avg_loss = torch.mean(torch.tensor(all_loss))
    avg_chamfer = torch.mean(torch.tensor(all_chamfer))
    avg_point_corr_dist = torch.mean(torch.tensor(all_point_corr_distance))

    logger.info(
        "[Val] Iter %04d | Loss %.6f  | CD %.6f  | Diff %.6f"
        % (it, avg_loss, avg_chamfer, avg_point_corr_dist)
    )
    writer.add_scalar("val/loss", avg_loss, it)
    writer.add_scalar("val/chamfer", avg_chamfer, it)
    writer.add_scalar("val/diff", avg_point_corr_dist, it)

    writer.flush()

    # scheduler.step(avg_chamfer)
    return loss, avg_chamfer, avg_point_corr_dist


# Main loop
logger.info("Start training...")
try:
    best_val_loss = float("inf")
    for epoch in range(args.max_epochs):
        train_for_one_epoch(epoch)

        loss, cd_loss, point_corr_dist = validate(epoch)
        opt_states = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        ckpt_mgr.save(model, args, loss, opt_states, name="last")
        if loss < best_val_loss:
            best_val_loss = loss
            ckpt_mgr.save(model, args, loss, opt_states, name="best")
        scheduler.step()

except KeyboardInterrupt:
    logger.info("Terminating...")
    wandb.finish()
