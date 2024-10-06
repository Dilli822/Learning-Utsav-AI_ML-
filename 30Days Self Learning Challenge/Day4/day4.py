

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# 1. Univariate Gaussian Distribution
def plot_univariate_gaussian():
    mu, sigma = 0, 1
    x = np.linspace(-4, 4, 1000)
    y = norm.pdf(x, mu, sigma)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Gaussian Distribution', color='blue')
    plt.title('Gaussian Distribution (Mean=0, SD=1)')
    plt.xlabel('X')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.legend()
    plt.show()

# 2. Bivariate Gaussian Distribution Setup
mu = np.array([0, 0])
cov = np.array([[1, 0.5], [0.5, 1]])
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
pos = np.empty(X1.shape + (2,))
pos[:, :, 0] = X1
pos[:, :, 1] = X2
rv = multivariate_normal(mu, cov)
Z = rv.pdf(pos)

def plot_joint_distribution():
    plt.figure(figsize=(10, 6))
    plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Probability Density')
    plt.title('Joint Probability Density Function')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def plot_marginal_distribution():
    marginal_X1 = norm.pdf(x1, mu[0], np.sqrt(cov[0][0]))
    plt.figure(figsize=(10, 6))
    plt.plot(x1, marginal_X1, label='Marginal Distribution of X1', color='orange')
    plt.title('Marginal Distribution (X1)')
    plt.xlabel('X1')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_conditional_distribution(x1_val):
    x2_conditional_mean = mu[1] + cov[0][1] / cov[0][0] * (x1_val - mu[0])
    x2_conditional_std = np.sqrt(cov[1][1] - (cov[0][1] ** 2) / cov[0][0])
    
    x2 = np.linspace(-3, 3, 100)
    conditional_pdf = norm.pdf(x2, loc=x2_conditional_mean, scale=x2_conditional_std)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x2, conditional_pdf, label=f'Conditional Distribution P(X2 | X1={x1_val})', color='green')
    plt.title('Conditional Distribution')
    plt.xlabel('X2')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_3d_distribution():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the 3D surface
    surf = ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
    
    # Customize the plot
    ax.set_title('3D Multivariate Gaussian Distribution')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Probability Density')
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.show()

def create_joint_probability_table():
    # Generate sample data
    n_samples = 1000
    samples = np.random.multivariate_normal(mu, cov, n_samples)
    
    # Create bins for discretization
    x1_bins = np.linspace(-2, 2, 5)
    x2_bins = np.linspace(-2, 2, 5)
    
    # Calculate joint probabilities
    hist, x_edges, y_edges = np.histogram2d(samples[:, 0], samples[:, 1], 
                                          bins=[x1_bins, x2_bins], density=True)
    
    # Create labels for the table
    x1_labels = [f'{x1_bins[i]:.1f} to {x1_bins[i+1]:.1f}' for i in range(len(x1_bins)-1)]
    x2_labels = [f'{x2_bins[i]:.1f} to {x2_bins[i+1]:.1f}' for i in range(len(x2_bins)-1)]
    
    # Create a DataFrame for better visualization
    joint_prob_df = pd.DataFrame(hist, index=x1_labels, columns=x2_labels)
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(joint_prob_df, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Joint Probability Distribution Table')
    plt.xlabel('X2')
    plt.ylabel('X1')
    plt.show()

# Execute all visualizations
if __name__ == "__main__":
    print("Generating all visualizations...")
    plot_univariate_gaussian()
    plot_joint_distribution()
    plot_marginal_distribution()
    plot_conditional_distribution(1)
    plot_3d_distribution()
    create_joint_probability_table()
    print("All visualizations complete!")


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

def create_3d_gaussian_with_marginals():
    # Set up the parameters
    mu = np.array([0, 0])
    cov = np.array([[1, 0.5], [0.5, 1]])
    
    # Create the grid
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)
    
    # Calculate the 2D Gaussian distribution
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    rv = multivariate_normal(mu, cov)
    Z = rv.pdf(pos)
    
    # Calculate marginal distributions
    dx = norm.pdf(x, mu[0], np.sqrt(cov[0,0]))
    dy = norm.pdf(y, mu[1], np.sqrt(cov[1,1]))
    
    # Create the figure with a specific size and 3D projection
    fig = plt.figure(figsize=(12, 8))
    
    # Set up the axes with gridspec
    gs = fig.add_gridspec(2, 2,  width_ratios=[3, 1], height_ratios=[1, 3],
                         left=0.1, right=0.9, bottom=0.1, top=0.9,
                         wspace=0.05, hspace=0.05)
    
    # Create the 3D scatter plot
    ax = fig.add_subplot(gs[1, 0], projection='3d')
    
    # Plot the 3D surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    # Add scatter points on the base for density visualization
    samples = np.random.multivariate_normal(mu, cov, 1000)
    ax.scatter(samples[:,0], samples[:,1], np.zeros_like(samples[:,0]), 
              c='black', alpha=0.1, s=1)
    
    # Customize the 3D plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Density')
    
    # Create the marginal distribution plots
    ax_top = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    
    # Plot marginal distributions
    ax_top.plot(x, dx, 'r-', lw=2)
    ax_right.plot(dy, y, 'b-', lw=2)
    
    # Fill the marginal distributions
    ax_top.fill_between(x, dx, alpha=0.3, color='red')
    ax_right.fill_betweenx(y, dy, alpha=0.3, color='blue')
    
    # Customize marginal plots
    ax_top.set_xticklabels([])
    ax_right.set_yticklabels([])
    
    # Set limits for all plots
    lim = 4
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax_top.set_xlim(-lim, lim)
    ax_right.set_ylim(-lim, lim)
    
    # Remove unnecessary spines
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    
    # Add a title
    plt.suptitle('Multivariate Gaussian Distribution with Marginals', y=0.95)
    
    # Add colorbar
    cb_ax = fig.add_axes([0.95, 0.1, 0.02, 0.3])
    fig.colorbar(surf, cax=cb_ax, label='Density')
    
    plt.show()

# Generate samples for scatter plot
def generate_samples(mu, cov, n_samples=1000):
    return np.random.multivariate_normal(mu, cov, n_samples)

# Execute the visualization
if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    create_3d_gaussian_with_marginals()