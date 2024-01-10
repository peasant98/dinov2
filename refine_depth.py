from dpt_module import open_image, DPT
import os

import open3d as o3d

import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import numpy as np

from scipy import ndimage
from scipy.optimize import minimize


def get_colmap_depth(root_dir, idx, scale=1000):
    depth_paths = os.listdir(root_dir)
    depth_paths_sorted = sorted(depth_paths)
    
    depth_valid_selected = depth_paths_sorted[idx]

    full_depth_image_path = os.path.join(root_dir, depth_valid_selected)
    print(full_depth_image_path)
    depth_image = cv2.imread(full_depth_image_path, cv2.IMREAD_UNCHANGED)/ scale
    depth_image = depth_image[:1099, :1799]
    return depth_image

def guided_filter_optimization(dense_depth, sparse_depth, radius=5, eps=0.1, lambda_sparse=0.5):
    """
    Optimize dense depth map using guided filter for geometry preservation and adherence to sparse depth points.

    :param dense_depth: Dense depth map (2D numpy array).
    :param sparse_depth: Sparse depth map (2D numpy array, same size, zeros where no depth is available).
    :param radius: Radius of the guided filter.
    :param eps: Regularization parameter of the guided filter.
    :param lambda_sparse: Weight for the sparse depth adherence.
    :return: Optimized dense depth map.
    """
    # Use the dense depth map as the guidance image for the filter
    guided_depth = cv2.ximgproc.guidedFilter(guide=dense_depth, src=dense_depth, radius=radius, eps=eps)

    # Integrate sparse depth information
    mask_sparse = sparse_depth > 0
    optimized_depth = guided_depth * (1 - lambda_sparse) + sparse_depth * lambda_sparse
    optimized_depth[~mask_sparse] = guided_depth[~mask_sparse]

    return optimized_depth


def optimize_depth(dense_depth, sparse_depth, alpha=0.01, iterations=1000):
    """
    Optimize the dense depth map to adhere to the sparse depth points while maintaining its original geometry.

    :param dense_depth: Initial dense depth map (2D NumPy array).
    :param sparse_depth: Sparse depth map (2D NumPy array, same size, zeros where no depth is available).
    :param alpha: Learning rate for the optimization.
    :param iterations: Number of iterations for the optimization process.
    :return: Optimized dense depth map.
    """
    optimized_depth = np.copy(dense_depth)
    sparse_mask = sparse_depth > 0

    for i in range(iterations):
        print('iteration', i)
        # Calculate gradient for adherence to sparse depth
        adherence_grad = np.zeros_like(dense_depth)
        adherence_grad[sparse_mask] = optimized_depth[sparse_mask] - sparse_depth[sparse_mask]

        adherence_grad *= 10
        
        # Calculate gradient for maintaining geometry (smoothness)
        geometry_grad = np.roll(optimized_depth, 1, axis=0) + \
                        np.roll(optimized_depth, -1, axis=0) + \
                        np.roll(optimized_depth, 1, axis=1) + \
                        np.roll(optimized_depth, -1, axis=1) - \
                        4 * optimized_depth
                        
        geometry_grad *= 0.01

        # Update the optimized depth map
        optimized_depth -= alpha * (adherence_grad + geometry_grad)

    return optimized_depth



def compute_gradient_loss(depth_map):
    """
    Compute the gradient loss (geometry consistency) for the depth map.

    :param depth_map: 2D numpy array of depth values.
    :return: Gradient loss and gradient of the loss w.r.t. the depth map.
    """
    grad_x = np.diff(depth_map, axis=1)
    grad_y = np.diff(depth_map, axis=0)

    # Compute the loss: mean of absolute gradients
    loss = np.mean(np.abs(grad_x)) + np.mean(np.abs(grad_y))

    # Compute the gradient of the loss
    grad_loss = np.zeros_like(depth_map)
    grad_loss[:, :-1] += np.sign(grad_x)  # Gradient w.r.t. x-axis differences
    grad_loss[:, 1:] -= np.sign(grad_x)   # Negative because of diff direction
    grad_loss[:-1, :] += np.sign(grad_y)  # Gradient w.r.t. y-axis differences
    grad_loss[1:, :] -= np.sign(grad_y)   # Negative because of diff direction

    return loss, grad_loss


def optimize_local_patch(dense_map, center_x, center_y, target_depth, patch_size=5, lambda_sparse=1.0, lambda_smooth=0.1):
    """
    Optimize a local patch of the dense map around a given sparse point.

    :param dense_map: Numpy array of the dense depth map.
    :param center_x, center_y: Coordinates of the sparse point.
    :param target_depth: Depth value at the sparse point.
    :param patch_size: Size of the square patch to optimize.
    :param lambda_sparse: Weight for the sparse depth point constraint.
    :param lambda_smooth: Weight for the smoothness constraint.
    :return: Optimized local patch.
    """
    start_x = max(center_x - patch_size // 2, 0)
    end_x = min(center_x + patch_size // 2 + 1, dense_map.shape[1])
    start_y = max(center_y - patch_size // 2, 0)
    end_y = min(center_y + patch_size // 2 + 1, dense_map.shape[0])

    original_patch = dense_map[start_y:end_y, start_x:end_x]

    # Define the objective function for the local patch
    def objective_function(local_patch_flat):
        local_patch = local_patch_flat.reshape(original_patch.shape)
        error = lambda_sparse * (local_patch[patch_size // 2, patch_size // 2] - target_depth) ** 2
        error += lambda_smooth * np.sum((local_patch - original_patch) ** 2)
        print(error)
        return error

    x0 = original_patch.flatten()
    result = minimize(objective_function, x0, method='L-BFGS-B')
    optimized_patch = result.x.reshape(original_patch.shape)

    return optimized_patch

def optimize_depth_map(dense_map, sparse_matrix, patch_size=5, lambda_sparse=1.0, lambda_smooth=0.1):
    optimized_map = np.copy(dense_map)
    y_coords, x_coords = np.nonzero(sparse_matrix)

    for y, x in zip(y_coords, x_coords):
        optimized_patch = optimize_local_patch(dense_map, x, y, sparse_matrix[y, x], patch_size, lambda_sparse, lambda_smooth)
        # Insert optimized patch back into the map
        start_x = max(x - patch_size // 2, 0)
        end_x = min(x + patch_size // 2 + 1, dense_map.shape[1])
        start_y = max(y - patch_size // 2, 0)
        end_y = min(y + patch_size // 2 + 1, dense_map.shape[0])
        optimized_map[start_y:end_y, start_x:end_x] = optimized_patch

    return optimized_map


def optimize_depthv2(dense_depth, sparse_depth, sparse_weight=0.1, geometry_weight = 0.2, iterations=1000, ignore_sparse=False):
    """
    Optimize the dense depth map to adhere to the sparse depth points while maintaining its original geometry.

    :param dense_depth: Initial dense depth map (2D NumPy array).
    :param sparse_depth: Sparse depth map (2D NumPy array, same size, zeros where no depth is available).
    :param alpha: Learning rate for the optimization.
    :param iterations: Number of iterations for the optimization process.
    :return: Optimized dense depth map.
    """
    optimized_depth = np.copy(dense_depth)
    sparse_mask = sparse_depth > 0

    for i in range(iterations):
        print('iteration', i)
        # Calculate gradient for adherence to sparse depth
        adherence_grad = np.zeros_like(dense_depth)
        adherence_grad[sparse_mask] = optimized_depth[sparse_mask] - sparse_depth[sparse_mask]
        adherence_grad *= sparse_weight

        # Calculate gradient for maintaining geometry (smoothness)
        # geometry_consistency_loss = compute_geometry_consistency_loss(optimized_depth)
        
        geometry_consistency_loss = np.roll(optimized_depth, 1, axis=0) + \
                        np.roll(optimized_depth, -1, axis=0) + \
                        np.roll(optimized_depth, 1, axis=1) + \
                        np.roll(optimized_depth, -1, axis=1) - \
                        4 * optimized_depth
                        
        geometry_grad = geometry_weight * geometry_consistency_loss
        
        if ignore_sparse:
            adherence_grad = 0

        # Update the optimized depth map
        optimized_depth -= (adherence_grad + geometry_grad)

    return optimized_depth

def remove_outliers(depth_image, blur_size=5, threshold=15):
    """
    Remove outliers from a depth image using Gaussian blur and thresholding.

    :param depth_image: Numpy array representing the depth image.
    :param blur_size: Size of the Gaussian blur filter.
    :param threshold: Threshold value to identify outliers.
    :return: Depth image with outliers removed.
    """
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(depth_image, (blur_size, blur_size), 0)

    # Calculate the difference
    difference = cv2.absdiff(depth_image, blurred)

    # Threshold to identify outliers
    _, outliers = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)

    # Option 1: Replace outliers with the corresponding values from the blurred image
    depth_image_filtered = np.where(outliers == 255, blurred, depth_image)

    # Option 2: Replace outliers with a fixed value (e.g., 0 or some other depth value)
    # depth_image_filtered = np.where(outliers == 255, 0, depth_image)

    return depth_image_filtered


def compute_geometry_consistency_loss(depth_map):
    """
    Compute geometry consistency loss for a depth map.

    :param depth_map: 2D numpy array of depth values.
    :return: The computed geometry consistency loss.
    """
    # Calculate gradients along x and y axes
    grad_x = np.abs(np.diff(depth_map, axis=1))
    grad_y = np.abs(np.diff(depth_map, axis=0))

    # Calculate the mean of the gradients as a simple form of geometry consistency loss
    loss_x = np.mean(grad_x)
    loss_y = np.mean(grad_y)

    return (loss_x + loss_y) / 2

def compute_scale_and_offset(sparse_depth, dense_depth):
    """
    Compute the scale factor and offset between sparse and dense depth maps.

    :param sparse_depth: Sparse depth map (2D numpy array).
    :param dense_depth: Dense depth map (2D numpy array, same size).
    :return: scale_factor, offset
    """
    # Mask to consider only the non-zero elements of the sparse depth map
    mask = sparse_depth > 0

    # Flattening the arrays and applying the mask
    sparse_depth_flat = sparse_depth[mask].flatten()
    dense_depth_flat = dense_depth[mask].flatten()

    # Performing linear regression
    A = np.vstack([dense_depth_flat, np.ones_like(dense_depth_flat)]).T
    scale_factor, offset = np.linalg.lstsq(A, sparse_depth_flat, rcond=None)[0]

    return scale_factor, offset

def create_point_cloud_from_depth(depth_map, intrinsics):
    """
    Convert a depth map to a point cloud object.

    :param depth_map: A 2D numpy array representing the depth map.
    :param intrinsics: The camera intrinsics as a dictionary with 'fx', 'fy', 'cx', 'cy'.
    :return: Open3D point cloud object
    """
    # Create Open3D depth image from the depth map
    depth_o3d = o3d.geometry.Image(depth_map.astype(np.float32))

    # Create intrinsic parameters object
    intrinsics_o3d = o3d.camera.PinholeCameraIntrinsic(
        depth_map.shape[1], depth_map.shape[0],
        intrinsics['fx'], intrinsics['fy'],
        intrinsics['cx'], intrinsics['cy']
    )

    # Create a point cloud from the depth image and intrinsics
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d, intrinsics_o3d
    )

    return pcd

def visualize_point_cloud(pcd):
    """
    Visualize a point cloud.

    :param pcd: Open3D point cloud object
    """
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    dpt_model = DPT()

    root_img_dir = 'bunny_imgs'
    root_colmap_depth_dir = 'colmap_depth'
    scale_factor = 1000
    
    output_depth_path = 'dense_depth_imgs'

    img_paths = sorted(os.listdir(root_img_dir))
    for idx, img_path in enumerate(img_paths):
        full_path = os.path.join(root_img_dir, img_path)
        
        image = open_image(full_path)

        dpt_depth = dpt_model(image, visualize=False)


        # get colmap depth
        colmap_depth = get_colmap_depth(root_colmap_depth_dir, idx, scale=1)
        # get scale factor
        
        scale, offset = compute_scale_and_offset(colmap_depth, dpt_depth)
        print(scale, offset)
        
        
        final_depthv1 = scale * dpt_depth + offset
        
        # final_depth = optimize_depthv2(final_depthv1, colmap_depth, sparse_weight=0.1, iterations=100, geometry_weight=0.000)
        # final_depth = final_depthv1
        # final_depth = remove_outliers(final_depth)
        
        # depth_image_normalized = (final_depth - np.min(final_depth)) / (np.max(final_depth) - np.min(final_depth))
        if False:
            # Apply a colormap for visualization
            # You can change 'plasma' to any other colormap (like 'viridis', 'magma', etc.)
            colored_image = cm.plasma(depth_image_normalized)

            # Display the colored depth image
            plt.imshow(colored_image)
            plt.colorbar()  # Adds a colorbar to interpret the values
            plt.title("Colored Depth Image")
            plt.show()
            
            camera_intrinsics = {'fx': 2500, 'fy': 2500, 'cx': 899.5, 'cy': 549.5}  # Replace with actual intrinsics
            pcd = create_point_cloud_from_depth(final_depth, camera_intrinsics)
            visualize_point_cloud(pcd)
            
            pcd = create_point_cloud_from_depth(colmap_depth, camera_intrinsics)
            visualize_point_cloud(pcd)
            
            pcd = create_point_cloud_from_depth(final_depthv1, camera_intrinsics)
            visualize_point_cloud(pcd)
            
            plt.figure(figsize=(12, 6))

            # Display the first depth image
            plt.subplot(1, 3, 1)  # (1 row, 2 columns, first subplot)
            plt.imshow(colmap_depth, cmap='viridis')
            plt.title('Colmap Depth')
            plt.axis('off')  # Turn off axis numbers

            # Display the second depth image
            plt.subplot(1, 3, 2)  # (1 row, 2 columns, second subplot)
            plt.imshow(dpt_depth, cmap='viridis')
            plt.title('DPT Depth')
            plt.axis('off')  # Turn off axis numbers
            
            plt.subplot(1, 3, 3)  # (1 row, 2 columns, second subplot)
            plt.imshow(final_depth, cmap='viridis')
            plt.title('Final Depth')
            plt.axis('off')  # Turn off axis numbers

            # Show the plot
            plt.show()
        
        final_depth_int = (scale_factor * colmap_depth).astype(np.uint16)
        
        if not os.path.exists('dense_depth_imgs'):
            os.mkdir('dense_depth_imgs')
        
        # name depth image without the "frame_0" and get last 4 digits
        img_path = img_path.split('.')[0][7:]
        
        cv2.imwrite(f'{output_depth_path}/{img_path}.png', final_depth_int)
        print('wrote depth image', f'{output_depth_path}/{img_path}.png')

        

        