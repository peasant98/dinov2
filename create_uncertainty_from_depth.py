import os

import open3d as o3d

import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import numpy as np

from scipy import ndimage
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter

def get_colmap_depth(root_dir, idx, scale=1000):
    depth_paths = os.listdir(root_dir)
    depth_paths_sorted = sorted(depth_paths)
    
    depth_valid_selected = depth_paths_sorted[idx]

    full_depth_image_path = os.path.join(root_dir, depth_valid_selected)
    print(full_depth_image_path)
    depth_image = cv2.imread(full_depth_image_path, cv2.IMREAD_UNCHANGED)/ scale
    depth_image = depth_image[:1099, :1799]
    return depth_image


def create_uncertainty_from_depth(depth, colmap_points, distance_weight=1.0):
    # uncertainty is higher for points that are further away
    # uncertainty is lower for points that are closer
    # implement this as a function of depth
    # uncertainty = 1 / depth
    uncertainty = (depth ** 2) * distance_weight
    return uncertainty


def create_uncertainty_map(depth_map, sparse_depth, distance_scale=0.01, sparse_scale=5):
    """
    Create an uncertainty map for a depth map.

    :param depth_map: Dense depth map as a 2D numpy array.
    :param sparse_depth: Sparse depth map as a 2D numpy array of the same size.
    :param distance_scale: Scaling factor for uncertainty based on depth.
    :param sparse_scale: Influence scale of sparse depth points on uncertainty.
    :return: Uncertainty map as a 2D numpy array.
    """
    # Step 1: Increase uncertainty with depth
    uncertainty_map = (depth_map ** 2) * distance_scale

    # Step 2: Decrease uncertainty near sparse depth points
    sparse_mask = sparse_depth > 0
    sparse_influence = gaussian_filter(sparse_mask.astype(float), sigma=sparse_scale)
    uncertainty_map *= (1 - sparse_influence)

    # Normalize the uncertainty map
    max_uncertainty = np.max(uncertainty_map)
    if max_uncertainty > 0:
        uncertainty_map /= max_uncertainty

    return uncertainty_map


def read_depth_image(path, scale=1000.0):
    depth = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    depth = depth / scale
    return depth


if __name__ == '__main__':
    depth_dir = 'dense_depth_imgs'
    depth_path = os.path.join(depth_dir, '0001.png')
    depth = read_depth_image(depth_path)
    
    colmap_depth = get_colmap_depth('colmap_depth', 0)
    
    uncertainty = create_uncertainty_from_depth(depth, None)
    
    # uncertainty = create_uncertainty_map
    
    # plot depth and uncertainty
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(depth, cmap='viridis')
    axs[1].imshow(colmap_depth, cmap='viridis')
    axs[2].imshow(uncertainty, cmap='viridis')
    plt.show()