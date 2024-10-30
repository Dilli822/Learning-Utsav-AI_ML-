import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define probabilities for each drop
prob_first_drop = {-1: 0.3, 1: 0.7}
prob_second_drop = {-1: 0.5, 1: 0.5}

# Possible final positions after two drops
positions = [-2, 0, 2]
prob_final_positions = {pos: 0 for pos in positions}

# Calculate final probabilities for each position
for first_step, p1 in prob_first_drop.items():
    for second_step, p2 in prob_second_drop.items():
        final_position = first_step + second_step
        prob_final_positions[final_position] += p1 * p2

# Initialize plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-3, 3)
ax.set_ylim(0, 3)
ax.axis('off')

# Draw initial and final nodes
ax.text(0, 2.5, "Start", ha="center", va="center", fontsize=12, fontweight='bold')
for pos, prob in prob_final_positions.items():
    ax.text(pos, 0.5, f"{prob:.2f}", ha="center", va="center", fontsize=12)

# Draw tree branches and probabilities
lines = []
texts = []

# Draw first set of branches (first drop)
for step, p in prob_first_drop.items():
    line, = ax.plot([0, step], [2.5, 1.5], 'b-', lw=2, alpha=0.6)
    lines.append(line)
    text = ax.text(step / 2, 2, f"{p:.2f}", ha="center", va="center", color="blue")
    texts.append(text)

# Draw second set of branches (second drop)
for first_step, p1 in prob_first_drop.items():
    for second_step, p2 in prob_second_drop.items():
        final_position = first_step + second_step
        line, = ax.plot([first_step, final_position], [1.5, 0.5], 'g-', lw=2, alpha=0.6)
        lines.append(line)
        text = ax.text((first_step + final_position) / 2, 1, f"{p1 * p2:.2f}", ha="center", va="center", color="green")
        texts.append(text)

# Animation function
def update(num):
    for i, line in enumerate(lines):
        line.set_alpha(0.1 if i != num else 0.8)
    for j, text in enumerate(texts):
        text.set_alpha(0.1 if j != num else 1)

# Run animation
ani = animation.FuncAnimation(fig, update, frames=len(lines), repeat=True, interval=800)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import convolve

# Define probability distributions for each drop
# f_x represents the probability for the first drop, and g_y for the second
def f_x(k):
    if k == -1:
        return 0.3
    elif k == 1:
        return 0.7
    else:
        return 0.0

def g_y(k):
    if k == -1:
        return 0.5
    elif k == 1:
        return 0.5
    else:
        return 0.0

# Define possible steps and compute probabilities
k_values = np.array([-1, 0, 1])
f_values = np.array([f_x(k) for k in k_values])
g_values = np.array([g_y(k) for k in k_values])

# Perform convolution
conv_result = convolve(f_values, g_values, mode='full')
final_positions = np.arange(-2, 3)

# Set up the figure and axes for plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Initial empty bar chart for the animated visualization
bars = ax.bar(final_positions, conv_result, color='purple', alpha=0.6)
ax.set_title("Combined Probability Distribution (After Two Drops)")
ax.set_xlabel("Final Position")
ax.set_ylabel("Probability")
ax.set_xticks(final_positions)
ax.set_ylim(0, max(conv_result) * 1.2)

# Function to animate the bars to show probability summing for each position
def animate(i):
    for j, bar in enumerate(bars):
        if j <= i:
            bar.set_alpha(1.0)  # Highlight bars up to the current position
        else:
            bar.set_alpha(0.2)  # Dim bars not yet reached
    return bars

# Create animation with updates on each frame
ani = FuncAnimation(fig, animate, frames=len(final_positions), interval=500, repeat=True)

plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import convolve

# Define probability distributions for each drop
def f_x(k):
    if k == -1:
        return 0.3
    elif k == 1:
        return 0.7
    else:
        return 0.0

def g_y(k):
    if k == -1:
        return 0.5
    elif k == 1:
        return 0.5
    else:
        return 0.0

# Define possible steps and compute probabilities
k_values = np.array([-1, 0, 1])
f_values = np.array([f_x(k) for k in k_values])
g_values = np.array([g_y(k) for k in k_values])

# Perform convolution
conv_result = convolve(f_values, g_values, mode='full')
final_positions = np.arange(-2, 3)

# Set up figure and axis for animation
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(0, max(conv_result) + 0.1)
ax.set_title("Animated Probability Distribution for Two Drops")
ax.set_xlabel("Final Position")
ax.set_ylabel("Probability")
ax.grid(True)

# Plot initialization
scatter_f, = ax.plot([], [], 'bo', ms=10, alpha=0.6, label="First Drop")
scatter_g, = ax.plot([], [], 'go', ms=10, alpha=0.6, label="Second Drop")
scatter_combined, = ax.plot([], [], 'ro', ms=10, alpha=0.6, label="Combined")

# Initialize the plot points
def init():
    scatter_f.set_data([], [])
    scatter_g.set_data([], [])
    scatter_combined.set_data([], [])
    return scatter_f, scatter_g, scatter_combined

# Animation function for each frame
def animate(i):
    if i < len(k_values):
        # First drop scatter animation
        scatter_f.set_data(k_values[:i+1], f_values[:i+1])
    elif i < 2 * len(k_values):
        # Second drop scatter animation
        scatter_g.set_data(k_values[:(i-len(k_values))+1], g_values[:(i-len(k_values))+1])
    else:
        # Combined convolution scatter animation
        scatter_combined.set_data(final_positions[:(i - 2 * len(k_values))+1], conv_result[:(i - 2 * len(k_values))+1])
    return scatter_f, scatter_g, scatter_combined

# Run animation
ani = FuncAnimation(fig, animate, frames=3 * len(k_values), init_func=init, interval=600, blit=True)

# Show animation
plt.legend()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Define probability distribution for a single drop
def f_x(k):
    if k == -1:
        return 0.3
    elif k == 1:
        return 0.7
    else:
        return 0.0

# Define the range of possible steps and initialize probability distribution
k_values = np.array([-1, 0, 1])
f_values = np.array([f_x(k) for k in k_values])

# Number of drops
n_drops = 100

# Perform convolution iteratively for n drops
combined_distribution = f_values
for _ in range(n_drops - 1):
    combined_distribution = convolve(combined_distribution, f_values, mode='full')

# Update the range of final positions to match combined distribution's length
final_positions = np.arange(-n_drops, n_drops + 1)

# Plot the final distribution
plt.figure(figsize=(10, 6))
plt.plot(final_positions, combined_distribution, color='purple', alpha=0.7, linewidth=2)
plt.title(f"Probability Distribution After {n_drops} Drops")
plt.xlabel("Final Position")
plt.ylabel("Probability")
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters for the normal distribution
mu = 0
sigma = 1

# Create figure and axis
fig, ax = plt.subplots()
x = np.linspace(-5, 5, 200)
line, = ax.plot(x, np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)), color='blue')

# Set axis limits and labels
ax.set_xlim(-5, 5)
ax.set_ylim(0, 1)
ax.set_title('Normal Distribution Animation')
ax.set_xlabel('X-axis')
ax.set_ylabel('Probability Density')

# Animation function
def update(frame):
    global sigma
    # Update sigma for animation effect
    sigma += 0.05
    # Recalculate y values based on new sigma
    y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    line.set_ydata(y)  # Update the y data of the line
    return line,

# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 100), blit=True, repeat=False)

# Show animation
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set up the figure and axis
fig, ax = plt.subplots()
x = np.linspace(-5, 5, 100)
line, = ax.plot(x, np.zeros_like(x), lw=2)

# Set the axis limits and labels
ax.set_xlim(-5, 5)
ax.set_ylim(0, 0.5)
ax.set_title("Animated Normal Distribution")
ax.set_xlabel("x")
ax.set_ylabel("Probability Density Function")

# Animation function to update the line
def update(frame):
    # Calculate the standard deviation and mean
    mean = 0
    std_dev = 1 + frame / 10  # Gradually increase the standard deviation
    y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    line.set_ydata(y)
    return line,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=30, blit=True, interval=200)

# Show the animation
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define parameters
n_frames = 100
initial_height = 10
drop_distance = 2

# Create figure and axes
fig, ax = plt.subplots()
ax.set_xlim(-5, 5)
ax.set_ylim(0, initial_height + 5)
ax.set_xlabel('Position')
ax.set_ylabel('Height')
ax.set_title('Object Drop Animation')

# Initialize the object position
object_pos = [0]
object_height = [initial_height]

# Create a line and point for the object
object_line, = ax.plot([], [], 'ro', markersize=10)
prob_text = ax.text(-4.5, initial_height + 1, '', fontsize=12)

def update(frame):
    ax.clear()
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, initial_height + 5)
    ax.set_xlabel('Position')
    ax.set_ylabel('Height')
    ax.set_title('Object Drop Animation')

    # First drop
    if frame < n_frames // 2:
        height = initial_height - (frame * (initial_height / (n_frames // 2)))
        object_pos = [0]
        object_height = [height]
        ax.plot(object_pos, object_height, 'ro', markersize=10)
        ax.plot([-5, 5], [0, 0], 'k-', lw=1)  # Surface line
        prob_text.set_text(f'First Drop: Height = {height:.2f}')
    else:
        # Second drop
        height = 0
        object_pos = np.random.uniform(-1, 1, size=5)
        object_height = [drop_distance] * len(object_pos)
        ax.plot(object_pos, object_height, 'ro', markersize=10)
        ax.plot([-5, 5], [0, 0], 'k-', lw=1)  # Surface line
        prob_text.set_text('Second Drop: Possible Positions')

        # Draw paths to ground
        for pos in object_pos:
            ax.plot([pos, pos], [drop_distance, 0], 'r--')

    return object_line, prob_text

# Create animation
ani = FuncAnimation(fig, update, frames=n_frames, blit=False, repeat=False)

# Display the animation
plt.show()
