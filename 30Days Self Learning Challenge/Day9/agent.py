import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from datetime import datetime

# Environment setup (10x10 grid with random obstacles)
grid_size = 10
grid = np.zeros((grid_size, grid_size))

# Define number of obstacles and their positions
num_obstacles = 10
obstacles = np.random.randint(0, grid_size, size=(num_obstacles, 2))

# Avoid placing the robot in an obstacle
agent_position = list(np.random.randint(0, grid_size, size=2))
for obs in obstacles:
    while np.array_equal(obs, agent_position):
        obs[:] = np.random.randint(0, grid_size, size=2)

# Function to check if the next position is valid (not an obstacle and within bounds)
def is_valid_move(position, obstacles):
    # Check if position matches any obstacle
    for obs in obstacles:
        if np.array_equal(position, obs):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] 🚫 Obstacle detected at position {position}!")
            return False
    
    # Check if position is within bounds
    if not (0 <= position[0] < grid_size and 0 <= position[1] < grid_size):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] ⚠️ Position {position} is out of bounds!")
        return False
        
    return True

# Initialize path and visited positions
path = []
visited = set()
current_position = list(agent_position)
path.append(current_position)

# Function to explore the environment
def explore_environment():
    global current_position
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] 🤖 Starting exploration from position {current_position}")
    
    while len(path) < grid_size * grid_size:  # Explore until all cells are visited
        # Possible moves (up, down, left, right)
        moves = [
            (current_position[0] + 1, current_position[1]),  # Move Down
            (current_position[0] - 1, current_position[1]),  # Move Up
            (current_position[0], current_position[1] + 1),  # Move Right
            (current_position[0], current_position[1] - 1)   # Move Left
        ]
        
        # Filter valid moves that haven't been visited
        valid_moves = []
        for move in moves:
            if is_valid_move(move, obstacles) and tuple(move) not in visited:
                valid_moves.append(move)
        
        if valid_moves:
            current_position = valid_moves[np.random.randint(len(valid_moves))]  # Randomly select a valid move
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] ✅ Moving to position {current_position}")
            path.append(current_position)
            visited.add(tuple(current_position))
        else:
            # No valid moves left, stop exploration
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] 🛑 No valid moves left! Ending exploration.")
            break

# Print initial obstacle positions
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
print(f"[{timestamp}] 🎯 Obstacles placed at positions:")
for i, obs in enumerate(obstacles):
    print(f"    Obstacle {i+1}: {obs}")

# Explore the environment
explore_environment()

# Rest of the visualization code remains the same...


# Function to draw the robotic agent
def draw_robot(ax, x, y):
    # Body
    body = patches.Circle((x, y), radius=0.3, color='blue')
    ax.add_patch(body)

    # Head
    head = patches.Circle((x, y + 0.4), radius=0.1, color='cyan')
    ax.add_patch(head)

    # Arms
    arm_left = patches.FancyArrow(x - 0.3, y, -0.2, 0, width=0.05, color='gray')
    arm_right = patches.FancyArrow(x + 0.3, y, 0.2, 0, width=0.05, color='gray')
    ax.add_patch(arm_left)
    ax.add_patch(arm_right)

    # Legs
    leg_left = patches.FancyArrow(x - 0.1, y - 0.35, -0.2, -0.2, width=0.05, color='gray')
    leg_right = patches.FancyArrow(x + 0.1, y - 0.35, 0.2, -0.2, width=0.05, color='gray')
    ax.add_patch(leg_left)
    ax.add_patch(leg_right)

    # Antennas
    antenna_left = patches.FancyArrow(x - 0.05, y + 0.45, -0.15, 0.3, width=0.02, color='red')
    antenna_right = patches.FancyArrow(x + 0.05, y + 0.45, 0.15, 0.3, width=0.02, color='red')
    ax.add_patch(antenna_left)
    ax.add_patch(antenna_right)

# Function to update the animation

# Function to update the animation
def update(frame):
    plt.clf()  # Clear the previous frame
    
    # Get current position
    current_position = path[frame % len(path)]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    # Print current robot position
    print(f"\n[{timestamp}] 🤖 Robot at position {current_position}")
    
    # Draw environment (obstacles)
    plt.imshow(grid, cmap='Greys', interpolation='none')
    
    # Check and print obstacle detections
    for obs in obstacles:
        plt.scatter(obs[1], obs[0], color='black', s=100)
        
        # Calculate distance to obstacle
        distance = abs(current_position[0] - obs[0]) + abs(current_position[1] - obs[1])
        
        # Print different messages based on proximity to obstacle
        if distance == 0:
            print(f"[{timestamp}] ⛔ COLLISION with obstacle at {obs}")
        elif distance == 1:
            print(f"[{timestamp}] 🚨 WARNING: Obstacle adjacent at {obs}")
        elif distance <= 2:
            print(f"[{timestamp}] ⚠️ CAUTION: Obstacle nearby at {obs}")

    # Draw the robot at the new position
    draw_robot(plt.gca(), current_position[1], current_position[0])
    
    # Print movement information
    if frame > 0:
        prev_position = path[(frame - 1) % len(path)]
        if prev_position != current_position:
            dx = current_position[1] - prev_position[1]
            dy = current_position[0] - prev_position[0]
            direction = ""
            if dx > 0: direction = "right"
            elif dx < 0: direction = "left"
            elif dy > 0: direction = "down"
            elif dy < 0: direction = "up"
            print(f"[{timestamp}] ➡️ Moving {direction}")
    
    # Add labels and details
    plt.title(f'AI Robotic Agent Exploring Environment - Step {frame}\nDetecting Obstacles in Real-time', 
              fontsize=14, fontweight='bold')
    plt.xlabel('X Coordinate (Environment Grid)', fontsize=12)
    plt.ylabel('Y Coordinate (Environment Grid)', fontsize=12)
    plt.grid(True)

# Create animation with longer interval to allow time for printing
fig = plt.figure(figsize=(8, 8))
ani = animation.FuncAnimation(fig, update, frames=len(path), interval=2000)  # 2 second delay

# Print initial setup
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
print(f"[{timestamp}] 🎯 Starting animation")
print(f"[{timestamp}] 🤖 Initial position: {path[0]}")
print(f"[{timestamp}] 📍 Number of obstacles: {len(obstacles)}")

plt.show()