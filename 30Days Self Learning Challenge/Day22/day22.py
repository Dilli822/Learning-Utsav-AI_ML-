import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the cost function: f(w) = w^2 (parabola)
def cost_function(w):
    return w**2

# Define the gradient of the cost function: f'(w) = 2*w
def gradient(w):
    return 2*w

# Gradient Descent Parameters
alpha = 0.1  # Learning rate
iterations = 20  # Number of iterations
w = 10  # Starting point (initial guess)
w_history = [w]  # To store the parameter values
cost_history = [cost_function(w)]  # To store the cost function values

# Gradient Descent Loop
for i in range(iterations):
    grad = gradient(w)
    w = w - alpha * grad  # Update the parameter using the gradient
    w_history.append(w)
    cost_history.append(cost_function(w))

# Prepare data for animation
w_vals = np.linspace(-10, 10, 100)  # Generate values for w
cost_vals = cost_function(w_vals)  # Calculate cost for each w value

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("Gradient Descent Visualization")
ax.set_xlabel("Parameter (w)")
ax.set_ylabel("Cost (f(w))")
ax.plot(w_vals, cost_vals, label="Cost Function: $f(w) = w^2$")
ax.legend()
ax.grid(True)

# Initialize the scatter plot for the steps
scatter = ax.scatter([], [], color="red")
line, = ax.plot([], [], color="red", linestyle='--')

# Function to initialize the animation
def init():
    scatter.set_offsets(np.empty((0, 2)))  # Set to empty 2D array
    line.set_data([], [])
    return scatter, line

# Function to update the animation
def update(frame):
    scatter.set_offsets([[w_history[frame], cost_history[frame]]])  # 2D array with shape (1, 2)
    line.set_data(w_history[:frame+1], cost_history[:frame+1])
    return scatter, line

# Create animation with slower interval (500 ms)
ani = animation.FuncAnimation(fig, update, frames=len(w_history), init_func=init, blit=True, interval=500, repeat=False)

# Show the animation
plt.show()
