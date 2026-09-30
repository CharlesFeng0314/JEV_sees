"""Depth clustering in the camera frame.

Object existence comes from geometry. Class names are attached later.
There is no table-plane or gripper assumption; the caller supplies intrinsics
and a working distance.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


def default_intrinsics(width: int, height: int) -> dict[str, float]:
    """Pinhole guess used only when the caller does not know the calibration."""

    focal = 0.5 * width / np.tan(np.deg2rad(30.0))
    return {"fx": float(focal), "fy": float(focal), "cx": width / 2.0, "cy": height / 2.0}


def depth_to_meters(depth: np.ndarray) -> np.ndarray:
    values = np.asarray(depth, dtype=np.float32)
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size and float(np.median(finite)) > 20.0:
        return values / 1000.0
    return values


def pixels_to_camera(
    xs: np.ndarray,
    ys: np.ndarray,
    depths_m: np.ndarray,
    intrinsics: dict[str, float],
) -> np.ndarray:
    x = (xs - float(intrinsics["cx"])) * depths_m / float(intrinsics["fx"])
    y = (ys - float(intrinsics["cy"])) * depths_m / float(intrinsics["fy"])
    return np.stack((x, y, depths_m), axis=-1)


def _find(parent: list[int], value: int) -> int:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def merge_nearby_clusters(points: np.ndarray, labels: np.ndarray, gap_m: float = 0.032) -> np.ndarray:
    """Merge DBSCAN fragments whose boxes nearly touch."""

    cluster_ids = sorted(int(value) for value in np.unique(labels) if value >= 0)
    if not cluster_ids:
        return labels
    bounds = {
        cluster_id: (
            points[labels == cluster_id].min(axis=0),
            points[labels == cluster_id].max(axis=0),
        )
        for cluster_id in cluster_ids
    }
    parent = list(range(max(cluster_ids) + 1))
    for index, left in enumerate(cluster_ids):
        for right in cluster_ids[index + 1 :]:
            minimum_left, maximum_left = bounds[left]
            minimum_right, maximum_right = bounds[right]
            gap = np.maximum(
                0.0,
                np.maximum(minimum_left - maximum_right, minimum_right - maximum_left),
            )
            if float(np.linalg.norm(gap)) < gap_m:
                left_root, right_root = _find(parent, left), _find(parent, right)
                if left_root != right_root:
                    parent[right_root] = left_root
    roots = {cluster_id: _find(parent, cluster_id) for cluster_id in cluster_ids}
    root_order = {root: index for index, root in enumerate(sorted(set(roots.values())))}
    merged = labels.copy()
    for cluster_id, root in roots.items():
        merged[labels == cluster_id] = root_order[root]
    return merged


def _dominant_plane_inliers(points: np.ndarray, distance_m: float = 0.015, min_fraction: float = 0.45) -> np.ndarray:
    """Mark points on the largest plane. A supporting surface is not an object."""

    inliers = np.zeros(len(points), dtype=bool)
    if len(points) < 80:
        return inliers
    rng = np.random.default_rng(0)
    best_count = 0
    best = inliers
    for _ in range(48):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        scale = float(np.linalg.norm(normal))
        if scale < 1e-8:
            continue
        normal = normal / scale
        offset = float(np.dot(normal, sample[0]))
        distances = np.abs(points @ normal - offset)
        chosen = distances < distance_m
        count = int(np.count_nonzero(chosen))
        if count > best_count:
            best_count = count
            best = chosen
    if best_count < min_fraction * len(points):
        return inliers
    return best


def cluster_depth(
    depth_m: np.ndarray,
    intrinsics: dict[str, float],
    *,
    near_m: float = 0.15,
    far_m: float = 2.0,
    pixel_stride: int = 4,
    eps_m: float = 0.045,
    min_samples: int = 5,
) -> list[np.ndarray]:
    """Return boolean masks for compact objects inside the working distance."""

    valid = np.isfinite(depth_m) & (depth_m > near_m) & (depth_m < far_m)
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return []
    sample_use = (xs % pixel_stride == 0) & (ys % pixel_stride == 0)
    sample_xs = xs[sample_use].astype(np.float64)
    sample_ys = ys[sample_use].astype(np.float64)
    sample_depth = depth_m[ys[sample_use], xs[sample_use]].astype(np.float64)
    sample_points = pixels_to_camera(sample_xs, sample_ys, sample_depth, intrinsics)
    if len(sample_points) < min_samples:
        return []
    kept = ~_dominant_plane_inliers(sample_points)
    sample_xs = sample_xs[kept]
    sample_ys = sample_ys[kept]
    sample_points = sample_points[kept]
    if len(sample_points) < min_samples:
        return []
    sample_labels = DBSCAN(eps=eps_m, min_samples=min_samples, algorithm="kd_tree", n_jobs=1).fit_predict(
        sample_points
    )
    sample_labels = merge_nearby_clusters(sample_points, sample_labels)
    good_labels = []
    for label in sorted(int(value) for value in np.unique(sample_labels) if value >= 0):
        points = sample_points[sample_labels == label]
        extent = points.max(axis=0) - points.min(axis=0)
        if len(points) < 12 or float(extent[2]) < 0.006 or float(np.max(extent)) > 0.55:
            continue
        good_labels.append(label)
    if not good_labels:
        return []
    selected = np.isin(sample_labels, good_labels)
    tree = cKDTree(sample_points[selected])
    kept_labels = sample_labels[selected]
    full_points = pixels_to_camera(
        xs.astype(np.float64),
        ys.astype(np.float64),
        depth_m[ys, xs].astype(np.float64),
        intrinsics,
    )
    distances, nearest = tree.query(full_points, k=1, workers=1)
    full_labels = np.full(len(full_points), -1, dtype=np.int32)
    close = distances <= max(0.045, eps_m * 1.35)
    full_labels[close] = kept_labels[nearest[close]]
    masks = []
    for label in good_labels:
        use = full_labels == label
        if int(np.count_nonzero(use)) < 30:
            continue
        mask = np.zeros(depth_m.shape, dtype=np.uint8)
        mask[ys[use], xs[use]] = 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        if int(np.count_nonzero(mask)) >= 30:
            masks.append(mask.astype(bool))
    return masks


def mask_facts(mask: np.ndarray, depth_m: np.ndarray, intrinsics: dict[str, float]) -> dict[str, Any] | None:
    valid = mask & np.isfinite(depth_m) & (depth_m > 0)
    ys, xs = np.nonzero(valid)
    if xs.size < 8:
        return None
    depths = depth_m[ys, xs].astype(np.float64)
    camera = pixels_to_camera(xs.astype(np.float64), ys.astype(np.float64), depths, intrinsics)
    median_depth = float(np.median(depths))
    near = np.abs(depths - median_depth) <= max(0.025, 0.04 * median_depth)
    source = camera[near] if int(np.count_nonzero(near)) >= 8 else camera
    center = np.median(source, axis=0)
    return {
        "area_px": int(xs.size),
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        "centroid_uv": [round(float(xs.mean()), 2), round(float(ys.mean()), 2)],
        "position_m": [round(float(value), 4) for value in center.tolist()],
    }
