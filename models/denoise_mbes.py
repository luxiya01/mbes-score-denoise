import torch
from torch import nn
import pytorch3d.ops
import numpy as np

from .feature import FeatureExtraction
from .score import ScoreNet


def get_random_indices(n, m):
    assert m < n
    return np.random.permutation(n)[:m]

def get_neighboring_indices(pnt_idx, K, pcl_length):
    """
    Given the seeds pnt_idx and the number of neighbors K,
    return the indices of the neighboring points.
    The seeding points will be placed in the middle of the neighboring points.
    Assumes that the given indices are NOT outside the range of the point cloud.
    """
    # Expand the indices
    idx_expand = np.arange(-(K - 1)//2, (K - 1)//2 + 1)
    idx_expand = pnt_idx[:, None] + idx_expand
    idx_expand = np.clip(idx_expand, 0, pcl_length - 1)
    return idx_expand


class DenoiseNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        # geometry
        self.num_train_points = args.num_train_points
        self.frame_knn = args.frame_knn
        self.num_clean_nbs = args.num_clean_nbs
        # score-matching
        self.dsm_sigma = args.dsm_sigma
        # networks
        self.feature_net = FeatureExtraction()
        self.score_net = ScoreNet(
            z_dim=self.feature_net.out_channels,
            dim=3,
            out_dim=3,
            hidden_size=args.score_net_hidden_dim,
            num_blocks=args.score_net_num_blocks,
        )

    def get_supervised_loss(self, pcl_noisy, pcl_clean, pcl_length):
        """
        Denoising score matching.
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
            pcl_clean:  Clean point clouds, (B, N, 3).
        """
        B, N, d = pcl_noisy.size()
        # pnt_idx = get_random_indices(N, self.num_train_points)
        K = self.frame_knn
        C = self.num_clean_nbs
        min_pcl_len = pcl_length.min().cpu()
        pnt_idx = np.random.permutation(
            np.arange(K, min_pcl_len - K)
        )[: self.num_train_points] # (n,)
        noisy_neighbor_idx = get_neighboring_indices(pnt_idx, K, min_pcl_len) # (n, K)
        #print(f'noisy neighbor_idx shape = {noisy_neighbor_idx.shape}')
        #print(f'noisy neighbor_idx = {noisy_neighbor_idx}')
        clean_neighbor_idx = get_neighboring_indices(noisy_neighbor_idx.flatten(),
                                                     C, min_pcl_len) # (nxK, C)
        clean_neighbor_idx = clean_neighbor_idx.reshape(-1, K, C) # (n, K, C)
        #print(f'clean neighbor_idx shape = {clean_neighbor_idx.shape}')
        #print(f'clean neighbor_idx = {clean_neighbor_idx}')

        # Feature extraction
        feat = self.feature_net(pcl_noisy)  # (B, N, F)
        feat = feat[:, pnt_idx, :]  # (B, n, F)
        F = feat.size(-1) # Feature dimension

        # Local frame construction
        noisy_frames = pcl_noisy[:, noisy_neighbor_idx, :] #(B, n, K, 3)
        clean_frames = pcl_clean[:, clean_neighbor_idx, :] #(B, n, K, C, 3)
        noisy_frame_centered = noisy_frames - pcl_noisy[:, pnt_idx, :].unsqueeze(
            2
        )  # (B, n, K, 3)

        noise_vecs = clean_frames - noisy_frames.unsqueeze(dim=3)  # (B, n, K, C, 3)
        noise_vecs = noise_vecs.mean(dim=3)  # (B, n, K, 3)
        grad_target = noise_vecs  # (B, n, K, 3)

        # Denoising score matching
        grad_pred = self.score_net(
            x=noisy_frame_centered.view(-1, K, d),
            c=feat.view(-1, F),
        ).reshape(
            B, len(pnt_idx), K, d
        )  # (B, n, K, 3)

        loss = (
            0.5
            * ((grad_target - grad_pred) ** 2.0 * (1.0 / self.dsm_sigma))
            .sum(dim=-1)
            .mean()
        )

        return loss  # , target, scores, noise_vecs

    def denoise_langevin_dynamics(
        self, pcl_noisy, step_size, denoise_knn=1, step_decay=0.95, num_steps=10
    ):
        """
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
        """
        B, N, d = pcl_noisy.size()
        with torch.no_grad():
            # Feature extraction
            self.feature_net.eval()
            feat = self.feature_net(pcl_noisy)  # (B, N, F)
            _, _, F = feat.size()

            # Trajectories
            traj = [pcl_noisy.clone().cpu()]
            pcl_next = pcl_noisy.clone()

            for step in range(num_steps):
                # Construct local frames
                seeds = np.arange(N)
                neighboring_idx = get_neighboring_indices(seeds, denoise_knn, N) # (N, K)
                # repeat neighboring_idx B times
                nn_idx = torch.LongTensor(neighboring_idx*np.ones((B, 1, 1))).view(
                    B, -1).to(pcl_noisy.device) # (B, N*K)
                frames = pcl_next[:, neighboring_idx, :]  # (B, N, K, 3)
                frames_centered = frames - pcl_noisy.unsqueeze(2) # (B, N, K, 3)

                # Predict gradients
                self.score_net.eval()
                grad_pred = self.score_net(
                    x=frames_centered.view(-1, denoise_knn, d), # (B*N, K, 3)
                    c=feat.view(-1, F), # (B*N, F)
                ).reshape(
                    B, -1, d
                )  # (B, N*K, 3)

                acc_grads = torch.zeros_like(pcl_noisy)
                acc_grads.scatter_add_(dim=1, index=nn_idx.unsqueeze(-1).expand_as(grad_pred), src=grad_pred)

                s = step_size * (step_decay**step)
                pcl_next += s * acc_grads
                traj.append(pcl_next.clone().cpu())
                # print(s)

        return pcl_next, traj
