import math
import torch
import pytorch3d.loss
import pytorch3d.structures
from pytorch3d.loss.point_mesh_distance import point_face_distance
from torch_cluster import fps


class FCLayer(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True, activation=None):
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

        if activation is None:
            self.activation = torch.nn.Identity()
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'elu':
            self.activation = torch.nn.ELU(alpha=1.0)
        elif activation == 'lrelu':
            self.activation = torch.nn.LeakyReLU(0.1)
        else:
            raise ValueError()

    def forward(self, x):
        return self.activation(self.linear(x))


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def truncated_normal_(tensor, mean=0, std=1, trunc_std=2):
    """
    Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def normalize_sphere(pc, radius=1.0):
    """
    Args:
        pc: A batch of point clouds, (B, N, 3).
    """
    ## Center
    p_max = pc.max(dim=-2, keepdim=True)[0]
    p_min = pc.min(dim=-2, keepdim=True)[0]
    center = (p_max + p_min) / 2    # (B, 1, 3)
    pc = pc - center
    ## Scale
    scale = (pc ** 2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0] / radius  # (B, N, 1)
    pc = pc / scale
    return pc, center, scale


def normalize_std(pc, std=1.0):
    """
    Args:
        pc: A batch of point clouds, (B, N, 3).
    """
    center = pc.mean(dim=-2, keepdim=True)   # (B, 1, 3)
    pc = pc - center
    scale = pc.view(pc.size(0), -1).std(dim=-1).view(pc.size(0), 1, 1) / std
    pc = pc / scale
    return pc, center, scale


def normalize_pcl(pc, center, scale):
    return (pc - center) / scale


def denormalize_pcl(pc, center, scale):
    return pc * scale + center


def chamfer_distance_unit_sphere(gen, ref, batch_reduction='mean', point_reduction='mean'):
    ref, center, scale = normalize_sphere(ref)
    gen = normalize_pcl(gen, center, scale)
    return pytorch3d.loss.chamfer_distance(gen, ref, batch_reduction=batch_reduction, point_reduction=point_reduction)


def farthest_point_sampling(pcls, num_pnts):
    """
    Args:
        pcls:  A batch of point clouds, (B, N, 3).
        num_pnts:  Target number of points.
    """
    ratio = 0.01 + num_pnts / pcls.size(1)
    sampled = []
    indices = []
    for i in range(pcls.size(0)):
        idx = fps(pcls[i], ratio=ratio, random_start=False)[:num_pnts]
        sampled.append(pcls[i:i+1, idx, :])
        indices.append(idx)
    sampled = torch.cat(sampled, dim=0)
    return sampled, indices


def point_mesh_bidir_distance_single_unit_sphere(pcl, verts, faces):
    """
    Args:
        pcl:    (N, 3).
        verts:  (M, 3).
        faces:  LongTensor, (T, 3).
    Returns:
        Squared pointwise distances, (N, ).
    """
    assert pcl.dim() == 2 and verts.dim() == 2 and faces.dim() == 2, 'Batch is not supported.'
    
    # Normalize mesh
    verts, center, scale = normalize_sphere(verts.unsqueeze(0))
    verts = verts[0]
    # Normalize pcl
    pcl = normalize_pcl(pcl.unsqueeze(0), center=center, scale=scale)
    pcl = pcl[0]

    # print('%.6f %.6f' % (verts.abs().max().item(), pcl.abs().max().item()))

    # Convert them to pytorch3d structures
    pcls = pytorch3d.structures.Pointclouds([pcl])
    meshes = pytorch3d.structures.Meshes([verts], [faces])
    return pytorch3d.loss.point_mesh_face_distance(meshes, pcls)


def pointwise_p2m_distance_normalized(pcl, verts, faces):
    assert pcl.dim() == 2 and verts.dim() == 2 and faces.dim() == 2, 'Batch is not supported.'
    
    # Normalize mesh
    verts, center, scale = normalize_sphere(verts.unsqueeze(0))
    verts = verts[0]
    # Normalize pcl
    pcl = normalize_pcl(pcl.unsqueeze(0), center=center, scale=scale)
    pcl = pcl[0]

    # Convert them to pytorch3d structures
    pcls = pytorch3d.structures.Pointclouds([pcl])
    meshes = pytorch3d.structures.Meshes([verts], [faces])

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # point to face distance: shape (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )
    return point_to_face


def hausdorff_distance_unit_sphere(gen, ref):
    """
    Args:
        gen:    (B, N, 3)
        ref:    (B, N, 3)
    Returns:
        (B, )
    """
    ref, center, scale = normalize_sphere(ref)
    gen = normalize_pcl(gen, center, scale)

    dists_ab, _, _ = pytorch3d.ops.knn_points(ref, gen, K=1)
    dists_ab = dists_ab[:,:,0].max(dim=1, keepdim=True)[0]  # (B, 1)
    # print(dists_ab)

    dists_ba, _, _ = pytorch3d.ops.knn_points(gen, ref, K=1)
    dists_ba = dists_ba[:,:,0].max(dim=1, keepdim=True)[0]  # (B, 1)
    # print(dists_ba)
    
    dists_hausdorff = torch.max(torch.cat([dists_ab, dists_ba], dim=1), dim=1)[0]

    return dists_hausdorff

def compute_mbes_denoising_metrics(gen, ref, valid_mask, denormalize=True, scale_xy=None, scale_z=None, center=None):
    if denormalize:
        for pcl in [gen, ref]:
            pcl *= scale_z
            pcl[:, :2] *= scale_xy
            pcl += center

    cd = pytorch3d.loss.chamfer_distance(gen.unsqueeze(0),
                                         ref.unsqueeze(0),
                                         point_reduction='mean')[0].item()

    gen_masked = gen.view(-1, 400, 3)[valid_mask]
    ref = ref.reshape(gen_masked.shape)
    z_diff = (gen_masked[:, 2] - ref[:, 2]).mean().item()
    z_abs_diff = (gen_masked[:, 2] - ref[:, 2]).abs().mean().item()
    z_rmse = torch.sqrt((gen_masked[:, 2] - ref[:, 2]).pow(2).mean()).item()
    return {'cd': cd, 'z_diff': z_diff, 'z_abs_diff': z_abs_diff, 'z_rmse': z_rmse}

def compute_outliers_iqr(grad, thresh=5.):
    quantiles = torch.quantile(grad, torch.tensor([.25, .5, .75]).to(grad.device))
    q1, median, q3 = quantiles
    iqr = q3 - q1
    # higher = q3 + 1.5 * iqr
    # lower = q1 - 1.5 * iqr
    higher = q3 + thresh * iqr
    lower = q1 - thresh * iqr
    # print(f'q1: {q1}, median: {median}, q3: {q3}, iqr: {iqr}, lower: {lower}, higher: {higher}')
    return torch.nonzero(torch.logical_or(grad > higher, grad < lower)).flatten()

def compute_mbes_outlier_rejection_metrics(gradients, gt_rejected, thresh=5.):
    grad_median = torch.median(gradients).item()
    # get indices where abs(gradients - grad_median) > std(gradients)*1.
    # grad_outliers = torch.nonzero(torch.abs(gradients - grad_median) > torch.std(gradients)*1.).flatten()
    grad_outliers = compute_outliers_iqr(gradients, thresh=thresh)
    # print(f'Number of outliers: {len(grad_outliers)}')

    # compute TP, FP, TN, FN using pred=grad_outliers and gt=gt_rejected
    def isin(a, b):
        return torch.sum(a[:, None] == b[None, :], dim=1) > 0
    TP = len(torch.nonzero(isin(grad_outliers, gt_rejected)).flatten())
    FP = len(grad_outliers) - TP
    FN = len(gt_rejected) - TP
    TN = gradients.flatten().shape[0] - (TP + FP + FN)
    return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}