# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# # Function to visualize scalars and vectors
# def visualize_scalars_vectors():
#     plt.figure(figsize=(10, 6))
    
#     # Define vectors
#     origin = np.array([0, 0])
#     v1 = np.array([1, 2])
#     v2 = np.array([3, 1])

#     # Plot vectors
#     plt.quiver(*origin, *v1, color='r', angles='xy', scale_units='xy', scale=1, label='Vector v1 (1,2)')
#     plt.quiver(*origin, *v2, color='b', angles='xy', scale_units='xy', scale=1, label='Vector v2 (3,1)')
    
#     plt.xlim(-1, 5)
#     plt.ylim(-1, 5)
#     plt.axhline(0, color='black', linewidth=0.5, ls='--')
#     plt.axvline(0, color='black', linewidth=0.5, ls='--')
#     plt.title('Scalars vs. Vectors Visualization')
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.grid()
#     plt.legend()
#     plt.show()

# # Function to visualize vector spaces
# def visualize_vector_spaces():
#     plt.figure(figsize=(8, 6))
    
#     # Define basis vectors
#     u = np.array([1, 0])
#     v = np.array([0, 1])
    
#     # Plot vectors
#     plt.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, color='r', label='Basis Vector u')
#     plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='b', label='Basis Vector v')
    
#     # Plot linear combinations
#     for a in np.linspace(-1, 1, 5):
#         for b in np.linspace(-1, 1, 5):
#             w = a * u + b * v
#             plt.quiver(0, 0, w[0], w[1], angles='xy', scale_units='xy', scale=1, color='g', alpha=0.3)
    
#     plt.xlim(-2, 2)
#     plt.ylim(-2, 2)
#     plt.axhline(0, color='black', linewidth=0.5, ls='--')
#     plt.axvline(0, color='black', linewidth=0.5, ls='--')
#     plt.title('Vector Spaces Visualization')
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.grid()
#     plt.legend()
#     plt.show()

# # Function to visualize linear transformations
# def visualize_linear_transformations():
#     plt.figure(figsize=(8, 6))
    
#     # Define original points
#     points = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])
#     transformation_matrix = np.array([[1, 0], [0, 2]])  # Scaling transformation

#     # Apply transformation
#     transformed_points = points @ transformation_matrix

#     # Plot original points
#     plt.scatter(points[:, 0], points[:, 1], color='r', label='Original Points', zorder=2)
    
#     # Plot transformed points
#     plt.scatter(transformed_points[:, 0], transformed_points[:, 1], color='b', label='Transformed Points', zorder=2)

#     # Draw arrows to show transformation
#     for point, transformed in zip(points, transformed_points):
#         plt.quiver(point[0], point[1], transformed[0] - point[0], transformed[1] - point[1], 
#                    angles='xy', scale_units='xy', scale=1, color='g', alpha=0.5)

#     plt.xlim(-2, 2)
#     plt.ylim(-3, 3)
#     plt.axhline(0, color='black', linewidth=0.5, ls='--')
#     plt.axvline(0, color='black', linewidth=0.5, ls='--')
#     plt.title('Linear Transformations Visualization')
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.grid()
#     plt.legend()
#     plt.show()

# # Function to visualize linear independence and dependence
# def visualize_linear_independence():
#     plt.figure(figsize=(8, 6))
    
#     # Define independent vectors
#     u1 = np.array([1, 1])
#     u2 = np.array([1, -1])

#     # Plot vectors
#     plt.quiver(0, 0, u1[0], u1[1], angles='xy', scale_units='xy', scale=1, color='r', label='Independent Vector 1')
#     plt.quiver(0, 0, u2[0], u2[1], angles='xy', scale_units='xy', scale=1, color='b', label='Independent Vector 2')
    
#     plt.xlim(-2, 2)
#     plt.ylim(-2, 2)
#     plt.axhline(0, color='black', linewidth=0.5, ls='--')
#     plt.axvline(0, color='black', linewidth=0.5, ls='--')
#     plt.title('Linear Independence Visualization')
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.grid()
#     plt.legend()
#     plt.show()

#     # Define dependent vectors
#     v1 = np.array([1, 2])
#     v2 = np.array([2, 4])  # v2 is a multiple of v1

#     # Plot dependent vectors
#     plt.figure(figsize=(8, 6))
#     plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='Dependent Vector 1')
#     plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='Dependent Vector 2 (2x v1)')
    
#     plt.xlim(-2, 3)
#     plt.ylim(-2, 6)
#     plt.axhline(0, color='black', linewidth=0.5, ls='--')
#     plt.axvline(0, color='black', linewidth=0.5, ls='--')
#     plt.title('Linear Dependence Visualization')
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.grid()
#     plt.legend()
#     plt.show()

# # Function to visualize polar and Cartesian coordinates
# def visualize_coordinates():
#     plt.figure(figsize=(8, 6))

#     # Cartesian coordinates
#     x = np.array([1, 0])
#     y = np.array([0, 1])
    
#     # Polar coordinates
#     r = np.sqrt(2)
#     theta = np.pi / 4

#     plt.quiver(0, 0, x[0], x[1], angles='xy', scale_units='xy', scale=1, color='r', label='Cartesian X')
#     plt.quiver(0, 0, y[0], y[1], angles='xy', scale_units='xy', scale=1, color='b', label='Cartesian Y')
#     plt.quiver(0, 0, r * np.cos(theta), r * np.sin(theta), angles='xy', scale_units='xy', scale=1, 
#                color='g', label='Polar Coordinate (r, θ)')

#     plt.xlim(-2, 2)
#     plt.ylim(-2, 2)
#     plt.axhline(0, color='black', linewidth=0.5, ls='--')
#     plt.axvline(0, color='black', linewidth=0.5, ls='--')
#     plt.title('Polar and Cartesian Coordinates Visualization')
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.grid()
#     plt.legend()
#     plt.show()

# # Function to visualize inner product
# def visualize_inner_product():
#     plt.figure(figsize=(8, 6))

#     # Define vectors
#     u = np.array([1, 2])
#     v = np.array([3, 4])

#     # Calculate inner product
#     inner_product = np.dot(u, v)

#     # Plot vectors
#     plt.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, color='r', label='Vector u (1, 2)')
#     plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='b', label='Vector v (3, 4)')

#     plt.xlim(-1, 5)
#     plt.ylim(-1, 5)
#     plt.axhline(0, color='black', linewidth=0.5, ls='--')
#     plt.axvline(0, color='black', linewidth=0.5, ls='--')
#     plt.title('Inner Product Visualization')
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.grid()
#     plt.legend()
#     plt.text(1, 4, f'Inner Product: {inner_product}', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
#     plt.show()

# # Function to visualize eigendecomposition
# # Function to visualize eigendecomposition
# def visualize_eigendecomposition():
#     plt.figure(figsize=(8, 6))

#     # Define matrix
#     matrix = np.array([[2, 1], [1, 2]])
#     eigvalues, eigvectors = np.linalg.eig(matrix)

#     # Plot eigenvectors
#     for i in range(len(eigvectors[0])):  # Iterate through each eigenvector
#         eigvector = eigvectors[:, i]
#         plt.quiver(0, 0, eigvector[0], eigvector[1], angles='xy', scale_units='xy', scale=1, 
#                    color='g', label=f'Eigenvector (Eigenvalue: {eigvalues[i]:.2f})')
    
#     # Plot original vectors
#     plt.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, color='r', label='Vector 1')
#     plt.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, color='b', label='Vector 2')

#     plt.xlim(-1, 2)
#     plt.ylim(-1, 2)
#     plt.axhline(0, color='black', linewidth=0.5, ls='--')
#     plt.axvline(0, color='black', linewidth=0.5, ls='--')
#     plt.title('Eigendecomposition Visualization')
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.grid()
#     plt.legend()
#     plt.show()


# # Function to visualize PCA
# def visualize_pca():
#     plt.figure(figsize=(8, 6))

#     # Create sample data
#     np.random.seed(0)
#     X = np.random.rand(10, 2) * 10

#     # Apply PCA
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X)

#     # Plot original data
#     plt.scatter(X[:, 0], X[:, 1], color='r', label='Original Data', zorder=2)

#     # Plot transformed data
#     plt.scatter(X_pca[:, 0], X_pca[:, 1], color='b', label='PCA Transformed Data', zorder=2)

#     plt.xlim(-5, 15)
#     plt.ylim(-5, 15)
#     plt.axhline(0, color='black', linewidth=0.5, ls='--')
#     plt.axvline(0, color='black', linewidth=0.5, ls='--')
#     plt.title('PCA Visualization')
#     plt.xlabel('X-axis')
#     plt.ylabel('Y-axis')
#     plt.grid()
#     plt.legend()
#     plt.show()

# # Call all visualization functions
# def main():
#     visualize_scalars_vectors()
#     visualize_vector_spaces()
#     visualize_linear_transformations()
#     visualize_linear_independence()
#     visualize_coordinates()
#     visualize_inner_product()
#     visualize_eigendecomposition()
#     visualize_pca()

# if __name__ == "__main__":
#     main()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set the style for all plots
plt.style.use('default')  # Using default style instead of seaborn

# Set global parameters for better visualization
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 12

def plot_vectors_and_scalars():
    """2D visualization of vectors vs scalars"""
    fig, ax = plt.subplots()
    
    # Plot vectors
    vector1 = np.array([2, 3])
    vector2 = np.array([1, 2])
    
    ax.quiver(0, 0, vector1[0], vector1[1], angles='xy', scale_units='xy', scale=1, color='blue', label='Vector 1')
    ax.quiver(0, 0, vector2[0], vector2[1], angles='xy', scale_units='xy', scale=1, color='red', label='Vector 2')
    
    # Plot scalar points
    ax.scatter([3], [3], color='green', s=100, label='Scalar Point')
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.grid(True)
    ax.set_title('Vectors vs Scalars Visualization')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.legend()
    plt.show()

def plot_3d_vectors():
    """3D visualization of vectors"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Create vectors
    vectors = [
        np.array([2, 1, 3]),
        np.array([-1, 2, 1]),
        np.array([1, -1, 2])
    ]
    colors = ['blue', 'red', 'green']
    
    # Plot each vector
    for vector, color in zip(vectors, colors):
        ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], 
                 color=color, label=f'Vector ({vector[0]}, {vector[1]}, {vector[2]})')
    
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Vector Space')
    ax.legend()
    plt.show()

def plot_linear_transformation():
    """Visualization of linear transformation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Create original points (square)
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    
    # Plot original shape
    ax1.plot(points[:, 0], points[:, 1], 'b-', label='Original')
    ax1.set_title('Original Shape')
    ax1.grid(True)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.legend()
    
    # Transform matrix (shear transformation)
    transform = np.array([[1, 0.5], [0, 1]])
    
    # Apply transformation
    transformed = np.dot(points, transform)
    
    # Plot transformed shape
    ax2.plot(transformed[:, 0], transformed[:, 1], 'r-', label='Transformed')
    ax2.set_title('After Linear Transformation')
    ax2.grid(True)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_polar_coordinates():
    """Visualization of polar coordinates"""
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # Create data
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.linspace(0, 2, 100)
    theta_grid, r_grid = np.meshgrid(theta, r)
    
    # Create contours
    ax.contour(theta_grid, r_grid, r_grid, levels=5)
    ax.set_title('Polar Coordinate System')
    plt.show()

def plot_vector_scaling():
    """Visualization of vector scaling"""
    fig, ax = plt.subplots()
    
    # Original vector
    vector = np.array([2, 1])
    ax.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', 
             scale=1, color='blue', label='Original Vector')
    
    # Scaled vectors
    scales = [0.5, 2]
    colors = ['red', 'green']
    
    for scale, color in zip(scales, colors):
        scaled = vector * scale
        ax.quiver(0, 0, scaled[0], scaled[1], angles='xy', scale_units='xy',
                 scale=1, color=color, label=f'Scaled by {scale}')
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_title('Vector Scaling')
    ax.legend()
    plt.show()

def plot_linear_dependence():
    """Visualization of linear dependence/independence"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Independent vectors
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    ax1.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, 
              color='blue', label='Vector 1')
    ax1.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1,
              color='red', label='Vector 2')
    ax1.set_title('Linearly Independent Vectors')
    ax1.grid(True)
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.legend()
    
    # Dependent vectors
    v3 = np.array([1, 1])
    v4 = np.array([2, 2])  # Multiple of v3
    ax2.quiver(0, 0, v3[0], v3[1], angles='xy', scale_units='xy', scale=1,
              color='blue', label='Vector 3')
    ax2.quiver(0, 0, v4[0], v4[1], angles='xy', scale_units='xy', scale=1,
              color='red', label='Vector 4 (2×Vector 3)')
    ax2.set_title('Linearly Dependent Vectors')
    ax2.grid(True)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Execute all visualizations
plot_vectors_and_scalars()
plot_3d_vectors()
plot_linear_transformation()
plot_polar_coordinates()
plot_vector_scaling()
plot_linear_dependence()

# Print information about the visualizations
print("\nVisualization Information:")
print("1. Vectors vs Scalars: Shows the difference between vector quantities (arrows) and scalar quantities (points)")
print("2. 3D Vectors: Demonstrates vectors in three-dimensional space")
print("3. Linear Transformation: Shows how matrices transform shapes in space")
print("4. Polar Coordinates: Displays the polar coordinate system")
print("5. Vector Scaling: Shows how vectors can be scaled by different factors")
print("6. Linear Dependence: Illustrates the concept of linear dependence and independence")