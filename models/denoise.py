import torch
from torch import nn
import pytorch3d.ops
import numpy as np

from .feature import FeatureExtraction
from .score import ScoreNet


def get_random_indices(n, m, excluded_idx=None):
    assert m < n
    if excluded_idx is None:
        return np.random.permutation(n)[:m]

    # Exclude some indices
    # If remaining indices are less than m, we need to sample with replacement
    excluded_idx = excluded_idx.cpu().numpy()
    all_indices = np.arange(n)
    indices = np.setdiff1d(all_indices, excluded_idx)
    replace = len(indices) < m
    return np.random.choice(indices, m, replace=replace)


class DenoiseNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        # geometry
        self.frame_knn = args.frame_knn
        self.num_train_points = args.num_train_points
        self.num_clean_nbs = args.num_clean_nbs
        if hasattr(args, 'num_selfsup_nbs'): self.num_selfsup_nbs = args.num_selfsup_nbs
        # score-matching
        self.dsm_sigma = args.dsm_sigma
        # networks
        self.feature_net = FeatureExtraction()
        self.score_net = ScoreNet(
            z_dim=self.feature_net.out_channels,
            dim=3, 
            out_dim=1,
            hidden_size=args.score_net_hidden_dim,
            num_blocks=args.score_net_num_blocks,
        )

    def get_supervised_loss(self, pcl_noisy, pcl_clean, rejected=None):
        """
        Denoising score matching.
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
            pcl_clean:  Clean point clouds, (B, M, 3). Usually, M is slightly greater than N.
            rejected:   Indices of rejected points in the noisy point clouds, (B, N_rejected).
        """
        B, N_noisy, N_clean, d = pcl_noisy.size(0), pcl_noisy.size(1), pcl_clean.size(1), pcl_noisy.size(2)
        n = self.num_train_points
        K = self.frame_knn
        C = self.num_clean_nbs
        # Avoid sampling outlier points
        pnt_idx = get_random_indices(N_noisy, self.num_train_points, excluded_idx=rejected)

        # Feature extraction
        feat = self.feature_net(pcl_noisy)  # (B, N, F)
        feat = feat[:,pnt_idx,:]  # (B, n, F)
        F = feat.size(-1)
        
        # # Local frame construction
        # _, _, frames = pytorch3d.ops.knn_points(pcl_noisy[:,pnt_idx,:], pcl_noisy, K=self.frame_knn, return_nn=True)  # (B, n, K, 3)
        # frames_centered = frames - pcl_noisy[:,pnt_idx,:].unsqueeze(2)   # (B, n, K, 3)

        # # Nearest clean points for each point in the local frame
        # # print(frames.size(), frames.view(-1, self.frame_knn, d).size())
        # _, _, clean_nbs = pytorch3d.ops.knn_points(
        #     frames.view(-1, self.frame_knn, d),    # (B*n, K, 3)
        #     pcl_clean.unsqueeze(1).repeat(1, len(pnt_idx), 1, 1).view(-1, N_clean, d),   # (B*n, M, 3)
        #     K=self.num_clean_nbs,
        #     return_nn=True,
        # )   # (B*n, K, C, 3)
        # clean_nbs = clean_nbs.view(B, len(pnt_idx), self.frame_knn, self.num_clean_nbs, d)  # (B, n, K, C, 3)

        # Local frame construction
        # neighbor_idx: (B, n, K), frames_xy: (B, n, K, 2)
        _, neighbor_idx, frames_xy = pytorch3d.ops.knn_points(pcl_noisy[:,pnt_idx,:2], pcl_noisy[..., :2], K=K, return_nn=True)
        # frames: (B, n, K, 1)
        frames = pcl_noisy[:, neighbor_idx.view(B, -1), :].reshape(B, n, K, 3) # (B, n, K, 3)
        frames_centered = frames - pcl_noisy[:,pnt_idx,:].reshape(B, n, 1, 3)   # (B, n, K, 3)
        # print(f'neighbor_idx: {neighbor_idx.size()}')
        # print(f'frames_xy: {frames_xy.size()}')
        # print(f'frames: {frames.size()}')
        # print(f'frames_centered: {frames_centered.size()}')

        # clean_nbs_idx: (B*n, K, C), clean_nbs_xy: (B*n, K, C, 2)
        _, clean_nbs_idx, clean_nbs_xy = pytorch3d.ops.knn_points(
            frames_xy.view(-1, K, 2),
            pcl_clean[..., :2].unsqueeze(1).repeat(1, len(pnt_idx), 1, 1).view(-1, N_clean, 2),   # (B*n, M, 2)
            K=self.num_clean_nbs,
            return_nn=True,
        )
        n, K, C = clean_nbs_idx.size()
        assert torch.eq(clean_nbs_idx.view(-1).view(B, -1, K, C), clean_nbs_idx).sum() == B * n * K * C
        # clean_nbs: (B, n, K, C, 3)
        clean_nbs = pcl_clean[:, clean_nbs_idx.view(-1), :].view(B, len(pnt_idx), self.frame_knn, self.num_clean_nbs, 3)
        # print(f'clean_nbs_idx: {clean_nbs_idx.size()}')
        # print(f'clean_nbs_xy: {clean_nbs_xy.size()}')
        # print(f'clean_nbs: {clean_nbs.size()}')

        # Noise vectors
        noise_vecs = frames.unsqueeze(dim=3) - clean_nbs  # (B, n, K, C, 3)
        noise_vecs = noise_vecs[..., -1].reshape(B, n, K, C, 1) # (B, n, K, C, 1)
        # print(f'noise_vecs: {noise_vecs.size()}')
        noise_vecs = noise_vecs.mean(dim=3)  # (B, n, K, 1)
        # print(f'noise_vecs: {noise_vecs.size()}')

        # Denoising score matching
        grad_pred = self.score_net(
            x = frames_centered.view(-1, self.frame_knn, 3),
            c = feat.view(-1, F),
        ).reshape(B, len(pnt_idx), self.frame_knn, 1)   # (B, n, K, 1)
        grad_target = - 1 * noise_vecs   # (B, n, K, 1)
        # print(f'grad_pred: {grad_pred.size()}')
        # print(f'grad_target: {grad_target.size()}')

        loss = 0.5 * ((grad_target - grad_pred) ** 2.0 * (1.0 / self.dsm_sigma)).sum(dim=-1).mean()
        
        return loss #, target, scores, noise_vecs

    def get_selfsupervised_loss(self, pcl_noisy):
        """
        Denoising score matching.
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
        """
        B, N_noisy, d = pcl_noisy.size()
        pnt_idx = get_random_indices(N_noisy, self.num_train_points)

        # Feature extraction
        feat = self.feature_net(pcl_noisy)  # (B, N, F)
        feat = feat[:,pnt_idx,:]  # (B, n, F)
        F = feat.size(-1)
        
        # Local frame construction
        _, _, frames = pytorch3d.ops.knn_points(pcl_noisy[:,pnt_idx,:], pcl_noisy, K=self.frame_knn, return_nn=True)  # (B, n, K, 3)
        frames_centered = frames - pcl_noisy[:,pnt_idx,:].unsqueeze(2)   # (B, n, K, 3)

        # Nearest points for each point in the local frame
        # print(frames.size(), frames.view(-1, self.frame_knn, d).size())
        _, _, selfsup_nbs = pytorch3d.ops.knn_points(
            frames.view(-1, self.frame_knn, d),    # (B*n, K, 3)
            pcl_noisy.unsqueeze(1).repeat(1, len(pnt_idx), 1, 1).view(-1, N_noisy, d),   # (B*n, M, 3)
            K=self.num_selfsup_nbs,
            return_nn=True,
        )   # (B*n, K, C, 3)
        selfsup_nbs = selfsup_nbs.view(B, len(pnt_idx), self.frame_knn, self.num_selfsup_nbs, d)  # (B, n, K, C, 3)

        # Noise vectors
        noise_vecs = frames.unsqueeze(dim=3) - selfsup_nbs  # (B, n, K, C, 3)
        noise_vecs = noise_vecs.mean(dim=3)  # (B, n, K, 3)

        # Denoising score matching
        grad_pred = self.score_net(
            x = frames_centered.view(-1, self.frame_knn, d),
            c = feat.view(-1, F),
        ).reshape(B, len(pnt_idx), self.frame_knn, d)   # (B, n, K, 3)
        grad_target = - 1 * noise_vecs   # (B, n, K, 3)

        loss = 0.5 * ((grad_target - grad_pred) ** 2.0 * (1.0 / self.dsm_sigma)).sum(dim=-1).mean()
        return loss #, target, scores, noise_vecs
  
    def denoise_langevin_dynamics(self, pcl_noisy, step_size, denoise_knn=4, step_decay=0.95, num_steps=30):
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

            init_grad = None
            for step in range(num_steps):
                # Construct local frames
                # _, nn_idx, frames = pytorch3d.ops.knn_points(pcl_noisy, pcl_next, K=denoise_knn, return_nn=True)   
                # frames_centered = frames - pcl_noisy.unsqueeze(2)   # (B, N, K, 3)
                # nn_idx = nn_idx.view(B, -1)    # (B, N*K)
                _, nn_idx, frames_xy = pytorch3d.ops.knn_points(pcl_noisy[..., :2], pcl_next[..., :2], K=denoise_knn, return_nn=True)
                nn_idx = nn_idx.view(B, -1)
                frames = torch.gather(pcl_noisy, 1, nn_idx.unsqueeze(-1).expand(-1, -1, 3)
                                      ).reshape(B, -1, denoise_knn, 3) # (B, n, K, 3)
                frames_centered = frames - pcl_next.unsqueeze(2)   # (B, N, K, 3)
                # frames_centered = frames - pcl_noisy.unsqueeze(2)   # (B, N, K, 3)
                # print(f'nn_idx: {nn_idx.size()}')
                # print(f'frames_xy: {frames_xy.size()}')
                # print(f'frames: {frames.size()}')
                # print(f'pcl_noisy: {pcl_noisy.size()}')
                # print(f'frames_centered: {frames_centered.size()}')

                # Predict gradients
                self.score_net.eval()
                grad_pred = self.score_net(
                    x=frames_centered.view(-1, denoise_knn, d),
                    c=feat.view(-1, F)
                ).reshape(B, -1, 1)   # (B, N*K, 1)

                grad_pred = grad_pred.view(B, N, denoise_knn, 1) # (B, N, K, 1)
                # acc_grads_mean = grad_pred.mean(dim=2) # (B, N, 1)
                acc_grads_median = grad_pred.median(dim=2).values
                acc_grads = acc_grads_median
                if init_grad is None:
                    init_grad = acc_grads.clone()

                # acc_grads = torch.zeros_like(pcl_noisy)
                # B, N, _ = pcl_noisy.size()
                # acc_grads = torch.zeros(B, N, 1).to(pcl_noisy.device)
                # acc_grads.scatter_add_(dim=1, index=nn_idx.unsqueeze(-1).expand_as(grad_pred), src=grad_pred)
                # print(f'grad_pred: {grad_pred.size()}')
                # print(f'acc_grads: {acc_grads.size()}')
                # print(f'acc_grads.squeeze(-1): {acc_grads.squeeze(-1).size()}')
                # print(f'pcl_next: {pcl_next.size()}')
                # print(f'pcl_next[:, :, 2]: {pcl_next[:, :, 2].size()}')

                s = step_size * (step_decay ** step)
                # pcl_next += s * acc_grads
                pcl_next[:, :, 2] -= s * acc_grads.squeeze(-1)
                traj.append(pcl_next.clone().cpu())
                # print(s)
            
        return pcl_next, traj, init_grad
