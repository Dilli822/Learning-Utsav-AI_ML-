import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Predicted probabilities
predicted_probs = np.array([0.1, 0.2, 0.5, 0.8, 0.9])

# Actual label y = 1 cost calculation
cost_y_1 = -np.log(predicted_probs)

# Actual label y = 0 cost calculation
cost_y_0 = -np.log(1 - predicted_probs)

# Setting up the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("Cost Function vs Predicted Probability in Logistic Regression")
ax.set_xlabel("Predicted Probability (hθ(x))")
ax.set_ylabel("Cost")
ax.set_xlim(0, 1)
ax.set_ylim(0, np.max(cost_y_1) + 1)
ax.grid(True)

# Initial empty lines for the animation
line1, = ax.plot([], [], label='Cost when y = 1', marker='o', color='green')
line2, = ax.plot([], [], label='Cost when y = 0', marker='o', color='red')

ax.legend()

# Animation initialization function
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2

# Animation update function
def update(frame):
    # Update line data
    line1.set_data(predicted_probs[:frame + 1], cost_y_1[:frame + 1])
    line2.set_data(predicted_probs[:frame + 1], cost_y_0[:frame + 1])
    return line1, line2

# Creating the animation
ani = FuncAnimation(fig, update, frames=len(predicted_probs)+1, init_func=init, blit=True, interval=1000)

plt.show()



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib.animation import FuncAnimation

# Generate synthetic data for two classes
np.random.seed(42)
X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.0, random_state=42)
y = y.astype(int)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot of the data points
scatter = ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0', alpha=0.6)
scatter = ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1', alpha=0.6)

# Define the decision boundary (a straight line)
# Adjust the slope and intercept to better separate the classes
x_values = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
m = 2  # slope (negative for better separation)
b = 5    # intercept (adjust to fit between classes)
y_values = m * x_values + b  # decision boundary line

# Initial empty line
line, = ax.plot([], [], color='green', label='Decision Boundary', linewidth=2)

# Setting up the limits and labels
ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
ax.set_title("Animated Decision Boundary")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.axhline(0, color='black', lw=0.5, ls='--')
ax.axvline(0, color='black', lw=0.5, ls='--')
ax.legend()
ax.grid()

# Animation function
def animate(i):
    line.set_data(x_values[:i], y_values[:i])  # Draw the line up to the i-th point
    return line,

# Create the animation
anim = FuncAnimation(fig, animate, frames=len(x_values), interval=50, blit=True)

# Show the animation
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generate circular data points
np.random.seed(42)

# Create points inside the circle (red)
r_inner = 0.5  # Radius for inner circle
n_inner = 50
theta_inner = np.random.uniform(0, 2 * np.pi, n_inner)
x_inner = r_inner * np.cos(theta_inner)
y_inner = r_inner * np.sin(theta_inner)

# Create points outside the circle (blue)
r_outer = 1.0  # Radius for outer circle
n_outer = 50
theta_outer = np.random.uniform(0, 2 * np.pi, n_outer)
x_outer = r_outer * np.cos(theta_outer) + np.random.normal(0, 0.1, n_outer)  # Adding some noise
y_outer = r_outer * np.sin(theta_outer) + np.random.normal(0, 0.1, n_outer)  # Adding some noise

# Combine the points into a single array
X = np.vstack((np.column_stack((x_inner, y_inner)), np.column_stack((x_outer, y_outer))))
y = np.array([0] * n_inner + [1] * n_outer)  # Labels: 0 for red, 1 for blue

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot of the data points
scatter = ax.scatter(x_inner, y_inner, color='red', label='Class 0 (Inside Circle)', alpha=0.6)
scatter = ax.scatter(x_outer, y_outer, color='blue', label='Class 1 (Outside Circle)', alpha=0.6)

# Define the circular decision boundary
theta = np.linspace(0, 2 * np.pi, 100)
x_circle = np.cos(theta)
y_circle = np.sin(theta)

# Initial empty line for the decision boundary
circle_boundary, = ax.plot([], [], color='green', label='Decision Boundary', linewidth=2)

# Setting up the limits and labels
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_title("Animated Circular Decision Boundary")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.axhline(0, color='black', lw=0.5, ls='--')
ax.axvline(0, color='black', lw=0.5, ls='--')
ax.set_aspect('equal', adjustable='box')
ax.legend()
ax.grid()

# Animation function
def animate(i):
    circle_boundary.set_data(x_circle[:i], y_circle[:i])  # Draw the circle up to the i-th point
    return circle_boundary,

# Create the animation
anim = FuncAnimation(fig, animate, frames=len(x_circle), interval=50, blit=True)

# Show the animation
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Generate circular data points
np.random.seed(42)

# Create points inside the circle (red)
r_inner = 0.5  # Radius for inner circle
n_inner = 50
theta_inner = np.random.uniform(0, 2 * np.pi, n_inner)
x_inner = r_inner * np.cos(theta_inner)
y_inner = r_inner * np.sin(theta_inner)

# Create points outside the circle (blue)
r_outer = 1.0  # Radius for outer circle
n_outer = 50
theta_outer = np.random.uniform(0, 2 * np.pi, n_outer)
x_outer = r_outer * np.cos(theta_outer) + np.random.normal(0, 0.1, n_outer)  # Adding some noise
y_outer = r_outer * np.sin(theta_outer) + np.random.normal(0, 0.1, n_outer)  # Adding some noise

# Combine the points into a single array
X = np.vstack((np.column_stack((x_inner, y_inner)), np.column_stack((x_outer, y_outer))))
y = np.array([0] * n_inner + [1] * n_outer)  # Labels: 0 for red, 1 for blue

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot of the data points
scatter = ax.scatter(x_inner, y_inner, color='red', label='Class 0 (Inside Circle)', alpha=0.6)
scatter = ax.scatter(x_outer, y_outer, color='blue', label='Class 1 (Outside Circle)', alpha=0.6)

# Define the circular decision boundary
theta = np.linspace(0, 2 * np.pi, 100)
x_circle = np.cos(theta)
y_circle = np.sin(theta)

# Initial empty line for the decision boundary
circle_boundary, = ax.plot([], [], color='green', label='Decision Boundary', linewidth=2)

# Setting up the limits and labels
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_title("Animated Circular Decision Boundary")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.axhline(0, color='black', lw=0.5, ls='--')
ax.axvline(0, color='black', lw=0.5, ls='--')
ax.set_aspect('equal', adjustable='box')
ax.legend()
ax.grid()

# Animation function
def animate(i):
    circle_boundary.set_data(x_circle[:i], y_circle[:i])  # Draw the circle up to the i-th point
    return circle_boundary,

# Create the animation
anim = FuncAnimation(fig, animate, frames=len(x_circle), interval=50, blit=True)

# Show the animation
plt.show()
