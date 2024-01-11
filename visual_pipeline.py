import os

from PIL import Image
import cv2
import numpy as np

from dpt_module import DPT
from utils import save_depth_image_matrix_as_npy
from zoe_depth import get_zoe_model
from create_uncertainty_from_depth import compute_uncertainty_map_with_edges

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_colmap_depth(root_dir, idx, scale=1000):
    depth_paths = os.listdir(root_dir)
    depth_paths_sorted = sorted(depth_paths)
    
    depth_valid_selected = depth_paths_sorted[idx]

    full_depth_image_path = os.path.join(root_dir, depth_valid_selected)
    depth_image = cv2.imread(full_depth_image_path, cv2.IMREAD_UNCHANGED)/ scale
    depth_image = depth_image[:1099, :1799]
    return depth_image


def open_image(image_path):
    image = Image.open(image_path)
    return image

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


class VisualPipeline:
    def __init__(self, root_img_dir, colmap_depth_dir='colmap_depth', output_depth_path='dense_depth', save_as_npy=True, scale_factor=1000):
        """Initializes the visual pipeline

        Args:
            root_img_dir (_type_): _description_
        """
        self.dpt_model = DPT()
        self.zoe_model = get_zoe_model()
        self.root_img_dir = root_img_dir
        self.colmap_depth_dir = colmap_depth_dir
        
        self.img_paths = sorted(os.listdir(self.root_img_dir))
        
        self.images = []
        self.colmap_depth_images = []
        
        self.get_images_and_colmap_depth_maps()
        
        self.output_depth_path = output_depth_path
        self.save_as_npy = save_as_npy
        self.scale_factor = scale_factor
        
        if not os.path.exists(self.output_depth_path):
            os.mkdir(self.output_depth_path)

        if not os.path.exists(f'{self.output_depth_path}_npy'):
            os.mkdir(f'{self.output_depth_path}_npy')

        if not os.path.exists(f'{self.output_depth_path}_uncertainty'):
            os.mkdir(f'{self.output_depth_path}_uncertainty')

        if not os.path.exists(f'{self.output_depth_path}_uncertainty_npy'):
            os.mkdir(f'{self.output_depth_path}_uncertainty_npy')
        
    def get_images_and_colmap_depth_maps(self, scale=1):
        for idx, img_path in enumerate(self.img_paths):
            full_path = os.path.join(self.root_img_dir, img_path)
            image = open_image(full_path)
            self.images.append(image)
            
            # get colmap depth images
            colmap_depth = get_colmap_depth(self.colmap_depth_dir, idx, scale=scale)
            self.colmap_depth_images.append(colmap_depth)
            
            print('Loaded:', full_path)
            
            
        
    def get_images(self):
        return self.images
    
    def get_colmap_depth_images(self):
        return self.colmap_depth_images
    
    def refine_depth_all_images(self, visualize=False):
        for i in range(len(self.images)):
            # predict depth from image
            predicted_depth = self.predict_depth_from_image(self.images[i])
            print(predicted_depth)
            # get colmap sparse(ish) depth
            colmap_depth = self.colmap_depth_images[i]
            
            
            # refine depth with a scale factor and offset
            refined_depth = self.refine_depth(predicted_depth, colmap_depth)
            if visualize:
                self.visualize(colmap_depth, predicted_depth, refined_depth)
            
            # get image path 
            img_path = self.img_paths[i].split('.')[0][7:]
            print(img_path)
            # # construct file path and save as depth image
            final_depth_int = (self.scale_factor * refined_depth).astype(np.uint16)
            cv2.imwrite(f'{self.output_depth_path}/{img_path}.png', final_depth_int)
            print(f'Saved depth image {self.output_depth_path}/{img_path}.png')
            # # save depth image matrix as npy
            if self.save_as_npy:
                file_path = os.path.join(f'{self.output_depth_path}_npy', f'{img_path}.npy')
                save_depth_image_matrix_as_npy(refined_depth, file_path)
            
            # compute uncertainty
            uncertainty = compute_uncertainty_map_with_edges(refined_depth, colmap_depth, edge_weight=1.0, distance_uncertainty_weight=0.02, proximity_weight=3.0)
            if visualize:
                self.visualize(colmap_depth, refined_depth, uncertainty)

            # create uncertainty depth image and npy file
                
            final_depth_uncertainty_int = (self.scale_factor * uncertainty).astype(np.uint16)
            cv2.imwrite(f'{self.output_depth_path}_uncertainty/{img_path}.png', final_depth_uncertainty_int)
            print(f'Saved depth uncertainty image {self.output_depth_path}_uncertainty/{img_path}.png')
                
            if self.save_as_npy:
                file_path = os.path.join(f'{self.output_depth_path}_uncertainty_npy', f'{img_path}.npy')
                save_depth_image_matrix_as_npy(uncertainty, file_path)

    def refine_depth(self, predicted_depth, colmap_depth):
            
        scale, offset = compute_scale_and_offset(colmap_depth, predicted_depth)
        final_depth = (scale * predicted_depth) + offset
        print('Scale:', scale)
        print('Offset:', offset)
        
        return final_depth
            
    def visualize(self, colmap_depth, predicted_depth, refined_depth):
        # Apply a colormap for visualization
        # You can change 'plasma' to any other colormap (like 'viridis', 'magma', etc.)
        
        plt.figure(figsize=(12, 6))

        # Display the first depth image
        plt.subplot(1, 3, 1)  # (1 row, 2 columns, first subplot)
        plt.imshow(colmap_depth, cmap='viridis')
        plt.title('Colmap Depth')
        plt.axis('off')  # Turn off axis numbers

        # Display the second depth image
        plt.subplot(1, 3, 2)  # (1 row, 2 columns, second subplot)
        plt.imshow(predicted_depth, cmap='viridis')
        plt.title('Predicted Depth')
        plt.axis('off')  # Turn off axis numbers
        
        plt.subplot(1, 3, 3)  # (1 row, 2 columns, second subplot)
        plt.imshow(refined_depth, cmap='viridis')
        plt.title('Final Depth')
        plt.axis('off')  # Turn off axis numbers

        # Show the plot
        plt.show()
        
        
    def predict_depth_from_image(self, image, model_type='zoe'):
        if model_type == 'zoe':
            depth = self.zoe_model.infer_pil(image)
        else:
            depth = self.dpt_model(image)
            
        return depth


if __name__ == '__main__':
    visual_pipeline = VisualPipeline(root_img_dir='bunny_imgs', colmap_depth_dir='colmap_depth')
    
    visual_pipeline.refine_depth_all_images(visualize=False)