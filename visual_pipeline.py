import os

from PIL import Image
import cv2
import numpy as np

from dpt_module import DPT
from transformers import pipeline

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
    print('Loading colmap depth image:', full_depth_image_path)
    depth_image = cv2.imread(full_depth_image_path, cv2.IMREAD_UNCHANGED).astype(np.float64)/ scale

    # depth_image = depth_image[:1099, :1799]
    
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
    def __init__(self, root_img_dir, colmap_depth_dir='colmap_blender_depth', output_depth_path='dense_blender_28_depth', save_as_npy=True, scale_factor=1000):
        """Initializes the visual pipeline

        Args:
            root_img_dir (_type_): _description_
        """
        self.dpt_model = DPT()
        self.zoe_model = get_zoe_model()
        self.depth_anything_model = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-large-hf")
        
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
            
        if not os.path.exists(f'{self.output_depth_path}_2_npy'):
            os.mkdir(f'{self.output_depth_path}_2_npy')
            
        if not os.path.exists(f'{self.output_depth_path}_4_npy'):
            os.mkdir(f'{self.output_depth_path}_4_npy')
        
        if not os.path.exists(f'{self.output_depth_path}_8_npy'):
            os.mkdir(f'{self.output_depth_path}_8_npy')

        if not os.path.exists(f'{self.output_depth_path}_uncertainty'):
            os.mkdir(f'{self.output_depth_path}_uncertainty')

        if not os.path.exists(f'{self.output_depth_path}_uncertainty_npy'):
            os.mkdir(f'{self.output_depth_path}_uncertainty_npy')
            
        if not os.path.exists(f'{self.output_depth_path}_uncertainty_2_npy'):
            os.mkdir(f'{self.output_depth_path}_uncertainty_2_npy')
        
        if not os.path.exists(f'{self.output_depth_path}_uncertainty_4_npy'):
            os.mkdir(f'{self.output_depth_path}_uncertainty_4_npy')
            
        if not os.path.exists(f'{self.output_depth_path}_uncertainty_8_npy'):
            os.mkdir(f'{self.output_depth_path}_uncertainty_8_npy')
        
    def get_images_and_colmap_depth_maps(self, scale=1000):
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
        errs = []
        for i in range(len(self.images)):
            # predict depth from image
            
            img_np = np.array(self.images[i])
            
            if len(img_np.shape) > 2 and img_np.shape[2] == 4:
                #convert the image from RGBA2RGB
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
            self.images[i] = Image.fromarray(img_np)
            
            predicted_depth = self.predict_depth_from_image(self.images[i])
            # get colmap sparse(ish) depth
            colmap_depth = self.colmap_depth_images[i]
            
            # refine depth with a scale factor and offset
            # predicted_depth = predicted_depth / -1
            refined_depth = self.refine_depth(predicted_depth, colmap_depth)
            # self.visualize(colmap_depth, predicted_depth, refined_depth)
            
            valid_locations = np.logical_and(~np.isnan(colmap_depth), colmap_depth != 0)

            # Compute the difference only at valid locations
            difference = np.abs(refined_depth - colmap_depth) * valid_locations

            # Calculate the average error (excluding invalid locations)
            average_error = np.sum(difference) / np.sum(valid_locations)
            errs.append(average_error)
            print(average_error)
            if visualize:
                self.visualize(predicted_depth, refined_depth, predicted_depth - refined_depth, labels=['Zoe Depth', 'Refined Depth', 'Depth Difference'])
            
            # get image path 
            img_path = self.img_paths[i].split('.')[0][7:]
            print(img_path)
            # # construct file path and save as depth image
            # final_depth_int = (self.scale_factor * refined_depth).astype(np.uint16)
            final_depth_int = (self.scale_factor * refined_depth).astype(np.uint16)
            
            
            depth_paths = os.listdir(self.colmap_depth_dir)
            depth_paths_sorted = sorted(depth_paths)
            
            depth_valid_selected = depth_paths_sorted[i]
                    
            cv2.imwrite(f'{self.output_depth_path}/{depth_valid_selected}', final_depth_int)
            print(f'Saved depth image {self.output_depth_path}/{depth_valid_selected}')
            # # save depth image matrix as npy
            if self.save_as_npy:
                file_path = os.path.join(f'{self.output_depth_path}_npy', f'{img_path}.npy')
                save_depth_image_matrix_as_npy(refined_depth, file_path)
                
                # resize refined depth to half
                refined_depth2 = cv2.resize(refined_depth, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                file_path = os.path.join(f'{self.output_depth_path}_2_npy', f'{img_path}.npy')
                save_depth_image_matrix_as_npy(refined_depth2, file_path)
                
                refined_depth4 = cv2.resize(refined_depth2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                file_path = os.path.join(f'{self.output_depth_path}_4_npy', f'{img_path}.npy')
                save_depth_image_matrix_as_npy(refined_depth4, file_path)
                
                refined_depth8 = cv2.resize(refined_depth4, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                file_path = os.path.join(f'{self.output_depth_path}_8_npy', f'{img_path}.npy')
                save_depth_image_matrix_as_npy(refined_depth8, file_path)
            
            # compute uncertainty
            uncertainty = compute_uncertainty_map_with_edges(refined_depth, colmap_depth, edge_weight=0, distance_uncertainty_weight=0.04, proximity_weight=50, depth_difference_weight=0, dilation_size=5)
            if visualize:
                self.visualize(colmap_depth, refined_depth, uncertainty, labels=['Colmap Depth', 'Refined Depth', 'Depth Uncertainty'])

            # create uncertainty depth image and npy file
                
            final_depth_uncertainty_int = (self.scale_factor * uncertainty).astype(np.uint16)
            cv2.imwrite(f'{self.output_depth_path}_uncertainty/{depth_valid_selected}', final_depth_uncertainty_int)
            print(f'Saved depth uncertainty image {self.output_depth_path}_uncertainty/{depth_valid_selected}')
                
            if self.save_as_npy:
                file_path = os.path.join(f'{self.output_depth_path}_uncertainty_npy', f'{img_path}.npy')
                save_depth_image_matrix_as_npy(uncertainty, file_path)
                
                uncertainty2 = cv2.resize(refined_depth, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                file_path = os.path.join(f'{self.output_depth_path}_uncertainty_2_npy', f'{img_path}.npy')
                save_depth_image_matrix_as_npy(uncertainty2, file_path)
                
                uncertainty4 = cv2.resize(refined_depth2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                file_path = os.path.join(f'{self.output_depth_path}_uncertainty_4_npy', f'{img_path}.npy')
                save_depth_image_matrix_as_npy(uncertainty4, file_path)
                
                uncertainty8 = cv2.resize(refined_depth4, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                file_path = os.path.join(f'{self.output_depth_path}_uncertainty_8_npy', f'{img_path}.npy')
                save_depth_image_matrix_as_npy(uncertainty8, file_path)
            print('average error:', np.mean(errs))
                
    def refine_depth(self, predicted_depth, colmap_depth):
            
        scale, offset = compute_scale_and_offset(colmap_depth, predicted_depth)
        final_depth = (scale * predicted_depth) + offset
        print('Scale:', scale)
        print('Offset:', offset)
        
        return final_depth
            
    def visualize(self, colmap_depth, predicted_depth, refined_depth, labels=['Colmap Depth', 'Predicted Depth', 'Refined Depth']):
        # Apply a colormap for visualization
        # You can change 'plasma' to any other colormap (like 'viridis', 'magma', etc.)
        
        plt.figure(figsize=(12, 6))

        # Display the first depth image
        plt.subplot(1, 3, 1)  # (1 row, 2 columns, first subplot)
        plt.imshow(colmap_depth, cmap='viridis')
        plt.title(labels[0])
        plt.axis('off')  # Turn off axis numbers

        # Display the second depth image
        plt.subplot(1, 3, 2)  # (1 row, 2 columns, second subplot)
        plt.imshow(predicted_depth, cmap='viridis')
        plt.title(labels[1])
        plt.axis('off')  # Turn off axis numbers
        
        plt.subplot(1, 3, 3)  # (1 row, 2 columns, second subplot)
        plt.imshow(refined_depth, cmap='viridis')
        plt.title(labels[2])
        plt.axis('off')  # Turn off axis numbers

        # Show the plot
        plt.show()
        
        
    def predict_depth_from_image(self, image, model_type='zoe'):
        if model_type == 'zoe':
            depth = self.zoe_model.infer_pil(image)
        elif model_type == 'depth_anything':
            depth = np.asarray(self.depth_anything_model(image)["depth"])
        else:
            depth = self.dpt_model(image)
            
        return depth


if __name__ == '__main__':
    visual_pipeline = VisualPipeline(root_img_dir='images_28', colmap_depth_dir='cmap_depth')
    
    # visual_pipeline = VisualPipeline(root_img_dir='bunny_square_images', colmap_depth_dir='bunny_square_sparse', output_depth_path='bunny_square_dense_depth')
    
    visual_pipeline.refine_depth_all_images(visualize=False)