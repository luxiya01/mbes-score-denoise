from functools import partial
from datasets.mbes import MBESDataset
from utils.transforms import NormalizeZ
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn import metrics
from tqdm.auto import tqdm
from collections import defaultdict
import pandas as pd
import argparse
from sklearn.neighbors import KDTree
from pykrige.ok import OrdinaryKriging


def o3d_pcl(pcl):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl)
    return pcd


def rejected_indices_to_mask(indices, length):
    """
    Args:
        indices: The indices of the kept points.
        length: The length of the original point cloud.
    Returns:
        mask: A boolean mask with the same length as the original point cloud.
              True = reject, False = keep.
    """
    mask = np.ones(length, dtype=bool)
    mask[indices] = False
    return mask


def remove_radius_outlier(pcl, nb_points=30, radius=0.03):
    pcl = o3d_pcl(pcl)
    cl, ind = pcl.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return rejected_indices_to_mask(ind, len(pcl.points))


def remove_statistical_outlier(pcl, nb_neighbors=30, std_ratio=1.5):
    pcl = o3d_pcl(pcl)
    cl, ind = pcl.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return rejected_indices_to_mask(ind, len(pcl.points))

def remove_outlier_ordinary_kriging(pcl, nb_neighbors=200, thresh=.02):
    pcl = pcl.reshape(-1, 3)
    kept_points = []
    X, Y, Z = pcl[:, 0], pcl[:, 1], pcl[:, 2]
    kdtree = KDTree(pcl[:, :2])
    _, ind = kdtree.query(pcl[:, :2], k=nb_neighbors+1)

    for i in range(X.shape[0]):
        neighbors_indices = ind[i][ind[i] != i] # remove the point itself
        neighbors = pcl[neighbors_indices]
        ok = OrdinaryKriging(neighbors[:, 0], neighbors[:, 1], neighbors[:, 2], variogram_model='linear')
        z_pred, _ = ok.execute('points', X[i], Y[i])
        if np.abs(z_pred[0] - Z[i])/z_pred[0] < thresh:
            kept_points.append(i)
    return rejected_indices_to_mask(kept_points, len(pcl))


def compute_binary_metrics(pred, gt):
    """
    Args:
        pred: The predicted mask.
        gt: The ground truth mask.
    Returns:
        A dictionary containing relevant binary classification metrics.
    """
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, np.logical_not(gt)).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()
    fn = np.logical_and(np.logical_not(pred), gt).sum()

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if precision + recall != 0
        else 0
    )

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def search_best_params(mbesdata, out_file="pcl_outlier_baseline.csv"):
    statistical_params = {
        "nb_neighbors": np.arange(10, 101, 10),
        "std_ratio": np.arange(0.5, 5.1, 0.5),
    }
    radius_params = {
        "nb_points": np.arange(10, 101, 10),
        "radius": np.arange(0.01, 0.101, 0.01),
    }
    # create a results dictionary to store the metrics for each parameter combination
    results = defaultdict(list)
    gt_rejection_mask = []

    for i, data in tqdm(enumerate(mbesdata), total=len(mbesdata)):
        rejected_mask = np.zeros(len(data["pcl_noisy"]), dtype=bool)
        rejected = data["rejected"].flatten()
        rejected_mask[rejected] = True
        gt_rejection_mask.append(rejected_mask)

        # iterate over all parameter combinations
        for nb_neighbors in statistical_params["nb_neighbors"]:
            for std_ratio in statistical_params["std_ratio"]:
                results[f"statistical_{nb_neighbors}_{std_ratio}"].append(
                    remove_statistical_outlier(
                        data["pcl_noisy"],
                        nb_neighbors=nb_neighbors,
                        std_ratio=std_ratio,
                    )
                )
        for nb_points in radius_params["nb_points"]:
            for radius in radius_params["radius"]:
                results[f"radius_{nb_points}_{radius}"].append(
                    remove_radius_outlier(
                        data["pcl_noisy"], nb_points=nb_points, radius=radius
                    )
                )

    gt_rejection_mask = np.concatenate(gt_rejection_mask).flatten()
    for k, v in results.items():
        results[k] = np.concatenate(v).flatten()

    df = pd.DataFrame(
        columns=[
            "method",
            "nb_neighbors",
            "std_ratio",
            "nb_points",
            "radius",
            "tp",
            "fp",
            "tn",
            "fn",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]
    )
    for k, v in results.items():
        if "statistical" in k:
            method, nb_neighbors, std_ratio = k.split("_")
            nb_points = np.nan
            radius = np.nan
        elif "radius" in k:
            method, nb_points, radius = k.split("_")
            nb_neighbors = np.nan
            std_ratio = np.nan
        metrics = compute_binary_metrics(v, gt_rejection_mask)
        df.loc[k] = {
            "method": method,
            "nb_neighbors": nb_neighbors,
            "std_ratio": std_ratio,
            "nb_points": nb_points,
            "radius": radius,
            **metrics,
        }
    df.to_csv(out_file, index=True)
    print(f"Total number of points: {len(gt_rejection_mask)}")
    print(f"Number of gt rejected points: {gt_rejection_mask.sum()}")
    return gt_rejection_mask

def outlier_removal(mbesdata, args):
    gt_rejection_mask = []
    pred_rejection_mask = []

    if args.method == "statistical":
        method = partial(remove_statistical_outlier, nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio)
    elif args.method == "radius":
        method = partial(remove_radius_outlier, nb_points=args.nb_points, radius=args.radius)
    elif args.method == "kriging":
        method = partial(remove_outlier_ordinary_kriging, nb_neighbors=args.nb_neighbors, thresh=args.thresh)
    else:
        raise ValueError(f"Method {args.method} not recognized."
                         f"Choose one of ['statistical', 'radius', 'kriging']")

    for i, data in tqdm(enumerate(mbesdata), total=len(mbesdata)):
        if i > 10:
            break
        rejected_mask = np.zeros(len(data["pcl_noisy"]), dtype=bool)
        rejected = data["rejected"].flatten()
        rejected_mask[rejected] = True
        gt_rejection_mask.append(rejected_mask)

        predicted_mask = method(data["pcl_noisy"])
        pred_rejection_mask.append(predicted_mask)
    gt_rejection_mask = np.concatenate(gt_rejection_mask).flatten()
    pred_rejection_mask = np.concatenate(pred_rejection_mask).flatten()
    metrics = compute_binary_metrics(pred_rejection_mask, gt_rejection_mask)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_path",
        type=str,
        default="/media/li/LaCie/Datasets/Gullmarsfjord-190618-svp+30/merged/patches_32pings_400_beams/",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="/media/li/LaCie/Datasets/Gullmarsfjord-190618-svp+30/draping_0.5m_/patches_32pings_400beams/",
    )
    parser.add_argument("--split", type=str, default="test_data.npz")
    parser.add_argument("--global_z_norm", type=bool, default=True)

    subparsers = parser.add_subparsers(dest="method", required=True)
    subparsers.add_parser("search_best_param", help="Search best parameters for statistical and radius outlier removal")

    radius = subparsers.add_parser("radius", help="Radius outlier removal method")
    radius.add_argument("--nb_points", type=int, default=30)
    radius.add_argument("--radius", type=float, default=0.03)

    statistical = subparsers.add_parser("statistical", help="Statistical outlier removal method")
    statistical.add_argument("--nb_neighbors", type=int, default=30)
    statistical.add_argument("--std_ratio", type=float, default=1.5)

    ordinary_kriging = subparsers.add_parser("kriging", help="Ordinary Kriging outlier removal method")
    ordinary_kriging.add_argument("--nb_neighbors", type=int, default=200)
    ordinary_kriging.add_argument("--thresh", type=float, default=0.02)

    return parser.parse_args()


def main():
    args = parse_args()

    z_scale = None
    if args.global_z_norm:
        z_scale = 28.82

    mbesdata = MBESDataset(
        args.raw_path,
        args.gt_path,
        split=args.split,
        transform=NormalizeZ(z_scale=z_scale),
    )
    if args.method == "search_best_param":
        norm_type = "global" if args.global_z_norm else "instance"
        search_best_params(mbesdata, out_file=f"pcl_outlier_baseline_{norm_type}_norm.csv")
    else:
        metrics = outlier_removal(mbesdata, args)
        print(metrics)


if __name__ == "__main__":
    main()
