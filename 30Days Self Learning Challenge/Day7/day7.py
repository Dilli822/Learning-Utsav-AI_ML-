import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# 1. 2D Matrix Transformation Animation
def matrix_transformation_2d_animation():
    # Define a transformation matrix
    A = np.array([[2, 1], [1, 2]])

    # Define a vector
    v = np.array([1, 1])

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    
    # Title and labels
    ax.set_title("2D Matrix Transformation Animation")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    # Initial plot of the vector
    original_vec, = ax.plot([0, v[0]], [0, v[1]], color='r', label='Original Vector (v)')
    transformed_vec, = ax.plot([], [], color='b', label='Transformed Vector (A*v)')

    # Update function for the animation
    def update(i):
        t = i / 100  # Time-based interpolation
        vec_t = (1 - t) * v + t * np.dot(A, v)  # Interpolate between original and transformed
        transformed_vec.set_data([0, vec_t[0]], [0, vec_t[1]])
        return transformed_vec,

    # Static parts of the plot
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.legend(loc='upper left')  # Place the legend
    plt.grid(True)

    # Create animation
    ani = FuncAnimation(fig, update, frames=100, blit=True)
    plt.show()

# 2. Eigenvector and Eigenvalue Animation (2D)
def eigenvector_animation():
    # Define a matrix
    A = np.array([[2, 1], [1, 2]])

    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    # Title and labels
    ax.set_title("2D Eigenvectors and Eigenvalues Transformation")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    # Plot original eigenvectors
    eig_vec_1, = ax.plot([0, eigenvectors[0, 0]], [0, eigenvectors[1, 0]], 'r', label=f'Eigenvector 1 (λ={eigenvalues[0]:.2f})')
    eig_vec_2, = ax.plot([0, eigenvectors[0, 1]], [0, eigenvectors[1, 1]], 'g', label=f'Eigenvector 2 (λ={eigenvalues[1]:.2f})')

    # Update function for the animation
    def update(i):
        # Apply matrix transformation to the eigenvectors over time
        t = i / 100
        vec1_t = (1 - t) * eigenvectors[:, 0] + t * np.dot(A, eigenvectors[:, 0]) * eigenvalues[0]
        vec2_t = (1 - t) * eigenvectors[:, 1] + t * np.dot(A, eigenvectors[:, 1]) * eigenvalues[1]

        eig_vec_1.set_data([0, vec1_t[0]], [0, vec1_t[1]])
        eig_vec_2.set_data([0, vec2_t[0]], [0, vec2_t[1]])

        return eig_vec_1, eig_vec_2

    # Static parts of the plot
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.legend(loc='upper left')
    plt.grid(True)

    # Create animation
    ani = FuncAnimation(fig, update, frames=100, blit=True)
    plt.show()

# 3. Eigen Decomposition Animation (3D)
def eigen_decomposition_animation_3d():
    # Define a 2x2 matrix
    A = np.array([[3, 1], [1, 3]])

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Create 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

    # Title and axis labels
    ax.set_title("3D Eigen Decomposition Animation")
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # Origin
    origin = np.array([0, 0, 0])

    # Plot eigenvectors
    eig_vec_1, = ax.plot([0, eigenvectors[0, 0] * eigenvalues[0]], [0, eigenvectors[1, 0] * eigenvalues[0]], [0, 0], color='r', label=f'Eigenvector 1 (λ={eigenvalues[0]:.2f})')
    eig_vec_2, = ax.plot([0, eigenvectors[0, 1] * eigenvalues[1]], [0, eigenvectors[1, 1] * eigenvalues[1]], [0, 0], color='b', label=f'Eigenvector 2 (λ={eigenvalues[1]:.2f})')

    # Update function for the animation
    def update(i):
        # Time progression for the transformation
        t = i / 100
        vec1_t = eigenvectors[:, 0] * (1 - t) + np.dot(A, eigenvectors[:, 0]) * eigenvalues[0] * t
        vec2_t = eigenvectors[:, 1] * (1 - t) + np.dot(A, eigenvectors[:, 1]) * eigenvalues[1] * t

        eig_vec_1.set_data([0, vec1_t[0]], [0, vec1_t[1]])
        eig_vec_2.set_data([0, vec2_t[0]], [0, vec2_t[1]])

        return eig_vec_1, eig_vec_2

    # Static parts of the plot
    ax.legend(loc='upper left')
    
    # Create 3D animation
    ani = FuncAnimation(fig, update, frames=100, blit=True)
    plt.show()

# Run all animations
matrix_transformation_2d_animation()   # 2D Matrix Transformation
eigenvector_animation()                # 2D Eigenvector Animation
eigen_decomposition_animation_3d()     # 3D Eigen Decomposition
