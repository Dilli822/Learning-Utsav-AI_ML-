
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skopt import gp_minimize
from skopt.space import Real
import random
import numpy as np

# Define a test function (e.g., a multi-modal function)
def test_function(x):
    return (x[0] - 2) ** 2 + np.sin(x[0] * 5) * 10  # Simple non-linear function

# Define the parameter space
param_space = [Real(-5, 5, name='x')]

# 1. Multiresolution Search: Coarse and Detailed
# Coarse Grid Search
coarse_x = np.linspace(-5, 5, 10)
coarse_y = [test_function([x]) for x in coarse_x]

# Detailed Grid Search
detailed_x = np.linspace(-2, 2, 20)  # Focused on a narrower range
detailed_y = [test_function([x]) for x in detailed_x]

# 2. Random Search (Random Grid)
random_x = [random.uniform(-5, 5) for _ in range(10)]
random_y = [test_function([x]) for x in random_x]

# 3. Bayesian Optimization
result = gp_minimize(test_function, param_space, n_calls=15, random_state=0)
bayes_x = [res[0] for res in result.x_iters]
bayes_y = result.func_vals

# Plot each search strategy
plt.figure(figsize=(12, 8))

# Plot the function
x = np.linspace(-5, 5, 100)
y = [test_function([xi]) for xi in x]
plt.plot(x, y, label="Function", color="gray", linestyle="--")

# Plot Multiresolution: Coarse and Detailed
plt.scatter(coarse_x, coarse_y, color="blue", label="Coarse Grid Search")
plt.scatter(detailed_x, detailed_y, color="green", label="Detailed Grid Search")

# Plot Random Search
plt.scatter(random_x, random_y, color="purple", label="Random Search")

# Plot Bayesian Optimization
plt.scatter(bayes_x, bayes_y, color="red", label="Bayesian Optimization", marker="x")

plt.xlabel("Parameter (x)")
plt.ylabel("Objective Function Value")
plt.legend()
plt.title("Comparison of Parameter Search Strategies")
plt.show()


# Generate function data for plotting
x = np.linspace(-5, 5, 100)
y = [test_function([xi]) for xi in x]

# Set up search points for each strategy
coarse_x = np.linspace(-5, 5, 10)
coarse_y = [test_function([x]) for x in coarse_x]

detailed_x = np.linspace(-2, 2, 20)
detailed_y = [test_function([x]) for x in detailed_x]

random_x = [random.uniform(-5, 5) for _ in range(10)]
random_y = [test_function([x]) for x in random_x]

# Perform Bayesian Optimization
result = gp_minimize(test_function, param_space, n_calls=15, random_state=0)
bayes_x = [res[0] for res in result.x_iters]
bayes_y = result.func_vals

# Initialize figure for animation
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, y, label="Function", color="gray", linestyle="--")
ax.set_xlabel("Parameter (x)")
ax.set_ylabel("Objective Function Value")
ax.set_title("Animated Parameter Search Strategies")
points, = ax.plot([], [], 'o', color='blue', label="Current Search Strategy")
legend = ax.legend()

# Animation function
def animate(i):
    # Reset plot at each frame
    points.set_data([], [])
    
    # Display points according to the frame
    if i < len(coarse_x):
        # Coarse grid search
        points.set_data(coarse_x[:i+1], coarse_y[:i+1])
        points.set_color("blue")
        points.set_label("Coarse Grid Search")
    elif i < len(coarse_x) + len(detailed_x):
        # Detailed grid search
        idx = i - len(coarse_x)
        points.set_data(detailed_x[:idx+1], detailed_y[:idx+1])
        points.set_color("green")
        points.set_label("Detailed Grid Search")
    elif i < len(coarse_x) + len(detailed_x) + len(random_x):
        # Random search
        idx = i - len(coarse_x) - len(detailed_x)
        points.set_data(random_x[:idx+1], random_y[:idx+1])
        points.set_color("purple")
        points.set_label("Random Search")
    else:
        # Bayesian Optimization
        idx = i - len(coarse_x) - len(detailed_x) - len(random_x)
        if idx < len(bayes_x):
            points.set_data(bayes_x[:idx+1], bayes_y[:idx+1])
            points.set_color("red")
            points.set_label("Bayesian Optimization")
    
    # Update legend to show current strategy
    legend = ax.legend()
    return points, legend

# Set total frames
frames = len(coarse_x) + len(detailed_x) + len(random_x) + len(bayes_x)

# Create animation
ani = FuncAnimation(fig, animate, frames=frames, blit=True, interval=500, repeat=False)

# Show animation
plt.show()



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

# Setting up random seed for reproducibility
np.random.seed(42)

# Parameters
num_layers = 5          # Number of layers in the model
num_epochs = 140        # Total number of epochs
epoch_range = np.arange(num_epochs)

# Generate synthetic activation values data
activation_data = [np.tanh(np.linspace(-2, 2, num_epochs) + layer) + 
                   0.2 * np.random.randn(num_epochs) for layer in range(num_layers)]

# Generate synthetic gradient distribution data
gradient_data = [np.random.normal(loc=0, scale=0.1 + 0.05 * layer, size=100) 
                 for layer in range(num_layers)]

# Set up the figure and axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle("Training Analysis: Activation Values and Gradient Distribution", fontsize=16)

# Colors for each layer
colors = sns.color_palette("viridis", num_layers)

# Initialize lines for activation values
activation_lines = [ax1.plot([], [], label=f'Layer {i+1}', color=colors[i])[0] for i in range(num_layers)]
ax1.set_xlim(0, num_epochs)
ax1.set_ylim(-1.5, 1.5)
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Activation Values")
ax1.legend(loc='upper right')
ax1.set_title("Top Plot: Activation Values Across Layers")

# Initialize histogram for gradient distribution
def init():
    for line in activation_lines:
        line.set_data([], [])
    return activation_lines

# Update function for animation
def update(epoch):
    # Update activation values
    for i, line in enumerate(activation_lines):
        line.set_data(epoch_range[:epoch], activation_data[i][:epoch])

    # Update gradient histogram
    ax2.clear()
    for i in range(num_layers):
        sns.histplot(gradient_data[i], kde=True, ax=ax2, color=colors[i], label=f'Layer {i+1}', element='step', fill=False)
    
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(0, 30)
    ax2.set_xlabel("Gradient Values")
    ax2.set_ylabel("Frequency")
    ax2.legend(loc='upper right')
    ax2.set_title("Bottom Plot: Gradient Distribution Across Layers")
    
    return activation_lines

# Animate
ani = animation.FuncAnimation(fig, update, frames=num_epochs, init_func=init, blit=False, repeat=False)

# Display the animation
plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
num_epochs = 100           # Number of epochs for visualization
num_features = 50          # Number of features in high-dimensional space
num_samples = 200          # Number of samples for t-SNE/UMAP

# Generate synthetic high-dimensional data representing the model's functions over epochs
high_dim_data_pretrained = np.array([np.random.normal(loc=0, scale=0.5, size=(num_samples, num_features)) 
                                     for _ in range(num_epochs)])
high_dim_data_not_pretrained = np.array([np.random.normal(loc=0, scale=1.0, size=(num_samples, num_features)) 
                                         for _ in range(num_epochs)])

# Function to apply t-SNE dimensionality reduction
def apply_tsne(data):
    tsne = TSNE(n_components=2, random_state=42)
    low_dim_data = tsne.fit_transform(data)
    return low_dim_data

# Prepare figure
fig = plt.figure(figsize=(12, 8))
fig.suptitle("Training Analysis: Low-Dimensional Representation and Sensitivity Analysis", fontsize=16)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

# Colors
colors_pretrained = sns.color_palette("cool", num_epochs)
colors_not_pretrained = sns.color_palette("hot", num_epochs)

# Plot Low-Dimensional Representations (2D Plot)
def plot_tsne(epoch):
    ax1.clear()
    ax1.set_title("Low-Dimensional Representation of Model's Functions")
    ax1.set_xlabel("t-SNE Component 1")
    ax1.set_ylabel("t-SNE Component 2")

    # Apply t-SNE on data of a specific epoch for each case
    tsne_data_pretrained = apply_tsne(high_dim_data_pretrained[epoch])
    tsne_data_not_pretrained = apply_tsne(high_dim_data_not_pretrained[epoch])
    
    ax1.scatter(tsne_data_pretrained[:, 0], tsne_data_pretrained[:, 1], 
                c=[colors_pretrained[epoch]], label="2 Layers with Pre-training", alpha=0.6)
    ax1.scatter(tsne_data_not_pretrained[:, 0], tsne_data_not_pretrained[:, 1], 
                c=[colors_not_pretrained[epoch]], label="2 Layers without Pre-training", alpha=0.6)

    ax1.legend(loc="upper right")

# Sensitivity Analysis (3D Plot)
def plot_sensitivity(epoch):
    ax2.clear()
    ax2.set_title("Weight Sensitivity Analysis")
    ax2.set_xlabel("Weight Variation")
    ax2.set_ylabel("Output Change")
    ax2.set_zlabel("Epoch")

    # Simulate sensitivity analysis by varying weights and measuring "output change"
    weight_variations = np.linspace(-1, 1, num_samples)
    output_changes_pretrained = np.sin(weight_variations + 0.05 * epoch)
    output_changes_not_pretrained = np.sin(weight_variations + 0.1 * epoch)

    ax2.scatter(weight_variations, output_changes_pretrained, epoch, color=colors_pretrained[epoch], 
                label="2 Layers with Pre-training", alpha=0.6)
    ax2.scatter(weight_variations, output_changes_not_pretrained, epoch, color=colors_not_pretrained[epoch], 
                label="2 Layers without Pre-training", alpha=0.6)

    ax2.legend(loc="upper right")

# Animation function
def animate(epoch):
    plot_tsne(epoch)
    plot_sensitivity(epoch)

# Create Animation
ani = animation.FuncAnimation(fig, animate, frames=num_epochs, interval=200, repeat=False)

plt.show()

