from dpt_module import open_image, DPT
import os

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from scipy import ndimage


def get_colmap_depth(root_dir, idx):
    depth_paths = os.listdir(root_dir)
    depth_paths_sorted = sorted(depth_paths)
    
    depth_valid_selected = depth_paths_sorted[idx]

    full_depth_image_path = os.path.join(root_dir, depth_valid_selected)
    
    depth_image = cv2.imread(full_depth_image_path, cv2.IMREAD_UNCHANGED)
    return depth_image

if __name__ == '__main__':
    dpt_model = DPT()

    root_img_dir = 'bunny_imgs'
    root_colmap_depth_dir = 'colmap_depth' 

    img_paths = os.listdir(root_img_dir)
    for idx, img_path in enumerate(img_paths):
        full_path = os.path.join(root_img_dir, img_path)
        
        image = open_image(full_path)

        dpt_depth = dpt_model(image, visualize=False)


        # get colmap depth
        colmap_depth = get_colmap_depth(root_colmap_depth_dir, idx)

        dense_depth_map = np.copy(dpt_depth)
        print(dpt_depth.shape)

        sparse_depth_map = np.copy(colmap_depth)

        non_zero_mask = sparse_depth_map > 0

        # Compute the nearest non-zero indices for each zero point
        distance, nearest_non_zero_indices = ndimage.distance_transform_edt(~non_zero_mask, return_indices=True)

        # Extract the row and column indices
        rows, cols = nearest_non_zero_indices

        # Map nearest non-zero sparse depth values and corresponding dense depth values
        nearest_sparse_depth = sparse_depth_map[rows, cols]
        nearest_dense_depth = dense_depth_map[rows, cols]

        # Compute ratios and fill in the sparse map
        ratios = nearest_sparse_depth / nearest_dense_depth
        filled_sparse_map = np.where(non_zero_mask, sparse_depth_map, ratios * nearest_dense_depth)



        # Visualization (displaying depth values of nearest non-zero points)
        plt.imshow(filled_sparse_map, cmap='viridis')
        plt.colorbar(label='Nearest Non-Zero Depth Value')
        plt.title('Nearest Non-Zero Depth Values for Zero Pixels')
        plt.show()


        plt.figure(figsize=(12, 6))

        # Display the first depth image
        plt.subplot(1, 2, 1)  # (1 row, 2 columns, first subplot)
        plt.imshow(dpt_depth, cmap='viridis')
        plt.title('Depth Image 1')
        plt.axis('off')  # Turn off axis numbers

        # Display the second depth image
        plt.subplot(1, 2, 2)  # (1 row, 2 columns, second subplot)
        plt.imshow(sparse_depth_map, cmap='viridis')
        plt.title('Depth Image 2')
        plt.axis('off')  # Turn off axis numbers

        # Show the plot
        plt.show()

        

        