from datasets.mbes import MBESDataset
from utils.transforms import NormalizeZ
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn import metrics
from tqdm.auto import tqdm
from collections import defaultdict
import pandas as pd


def o3d_pcl(pcl):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl)
    return pcd


def rejected_indices_to_mask(indices, length):
    """
    Args:
        indices: The indices of the rejected points.
        length: The length of the original point cloud.
    Returns:
        mask: A boolean mask with the same length as the original point cloud.
              True = reject, False = keep.
    """
    mask = np.ones(length, dtype=bool)
    mask[indices] = False
    return mask


def remove_radius_outlier(pcl, nb_points=20, radius=0.03):
    pcl = o3d_pcl(pcl)
    cl, ind = pcl.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return rejected_indices_to_mask(ind, len(pcl.points))


def remove_statistical_outlier(pcl, nb_neighbors=30, std_ratio=1.0):
    pcl = o3d_pcl(pcl)
    cl, ind = pcl.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return rejected_indices_to_mask(ind, len(pcl.points))


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
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

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

def main():
    split = 'test_data.npz'
    raw_path = f"/media/li/LaCie/Datasets/Gullmarsfjord-190618-svp+30/merged/patches_32pings_400_beams/{split}/"
    gt_path = f"/media/li/LaCie/Datasets/Gullmarsfjord-190618-svp+30/draping_0.5m_/patches_32pings_400beams/{split}/"
    mbesdata = MBESDataset(raw_path, gt_path, split="", transform=NormalizeZ())

    gt_rejection_mask = []
    statistical_params = {
        "nb_neighbors": np.arange(10, 100, 10),
        "std_ratio": np.arange(.5, 5, .5),
    }
    radius_params = {
        "nb_points": np.arange(10, 100, 10),
        "radius": np.arange(.01, .1, .01),
    }
    # create a results dictionary to store the metrics for each parameter combination
    results = defaultdict(list)

    for i, data in tqdm(enumerate(mbesdata), total=len(mbesdata)):
        gt_rejection_mask.append(data["rejection_mask"].flatten())

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
    df.to_csv("pcl_outlier_removal_baselines.csv", index=True)

    print(f"Total number of points: {len(gt_rejection_mask)}")
    print(f"Number of gt rejected points: {gt_rejection_mask.sum()}")


if __name__ == "__main__":
    main()