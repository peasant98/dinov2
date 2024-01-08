import numpy as np
from scipy.optimize import minimize

# Example dense depth map (replace with your data)
dense_depth_map = np.random.rand(480, 640) * 10  # Some synthetic data for demonstration

# Example sparse depth map (replace with your data)
sparse_depth_map = np.zeros((480, 640))
sparse_depth_map[100, 150] = 3.5
sparse_depth_map[200, 400] = 5.7
sparse_depth_map[350, 300] = 2.2

# Define a cost function
def cost_function(adjusted_depth_map_flat, dense_depth_map_flat, sparse_depth_map_flat, lambda_weight):
    # Reshape the flattened adjusted depth map back to its original shape
    adjusted_depth_map = adjusted_depth_map_flat.reshape(dense_depth_map.shape)

    # Term 1: Deviation from the original dense depth map
    term_1 = np.sum((adjusted_depth_map - dense_depth_map_flat) ** 2)

    # Term 2: Deviation from sparse depth points
    sparse_mask = sparse_depth_map_flat > 0
    term_2 = np.sum((adjusted_depth_map_flat[sparse_mask] - sparse_depth_map_flat[sparse_mask]) ** 2)

    return term_1 + lambda_weight * term_2

# Flatten the depth maps for optimization
dense_depth_map_flat = dense_depth_map.flatten()
sparse_depth_map_flat = sparse_depth_map.flatten()

# Lambda weight to balance the two terms
lambda_weight = 1.0

# Initial guess (start with the original dense depth map)
initial_guess = dense_depth_map_flat

# Run the optimization
result = minimize(cost_function, initial_guess, args=(dense_depth_map_flat, sparse_depth_map_flat, lambda_weight), method='L-BFGS-B')

# Reshape the result back to the original shape
optimized_depth_map = result.x.reshape(dense_depth_map.shape)

# Show the optimized depth map (can use matplotlib or other visualization tools)
# For example:
# import matplotlib.pyplot as plt
# plt.imshow(optimized_depth_map, cmap='gray')
# plt.colorbar()
# plt.title("Optimized Depth Map")
# plt.show()
