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
        #self.num_train_points = args.num_train_points
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
