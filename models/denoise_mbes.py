import torch
from torch import nn
import pytorch3d.ops
import numpy as np

from .feature import FeatureExtraction
from .score import ScoreNet


def get_random_indices(n, m):
    assert m < n
    return np.random.permutation(n)[:m]


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
        B, N, _ = pcl_noisy.size()
        # Feature extraction
        feat = self.feature_net(pcl_noisy)  # (B, N, F)
        F = feat.size(-1)

        noise_vecs = pcl_noisy - pcl_clean # (B, N, 3)
        grad_target = - 1 * noise_vecs   # (B, n, 3)

        # Denoising score matching
        grad_pred = self.score_net(
            x = pcl_noisy.view(-1, 1, 3),
            c = feat.view(-1, F),
        ).reshape(B, N, 3)   # (B, N, 3)

        # Mask out padded points
        idx_mask = torch.arange(N).expand(B, N).to(pcl_length.device) < pcl_length.unsqueeze(1)
        grad_pred = grad_pred.masked_select(idx_mask.unsqueeze(-1)).view(-1, 3)
        grad_target = grad_target.masked_select(idx_mask.unsqueeze(-1)).view(-1, 3)

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
  
    def denoise_langevin_dynamics(self, pcl_noisy, step_size, step_decay=0.95, num_steps=30):
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
                # Predict gradients
                self.score_net.eval()
                grad_pred = self.score_net(
                    x=pcl_noisy.view(-1, 1, d),
                    c=feat.view(-1, F)
                ).reshape(B, -1, d)   # (B, N*K, 3)

                s = step_size * (step_decay ** step)
                pcl_next += s * grad_pred
                traj.append(pcl_next.clone().cpu())
                # print(s)
            
        return pcl_next, traj
