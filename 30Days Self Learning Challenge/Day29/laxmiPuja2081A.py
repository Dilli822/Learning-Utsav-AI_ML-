import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import colorsys
import random
import math

# Load the image
image_path = 'th.png'  # Replace with your image path
image = Image.open(image_path)

# Convert image to grayscale
image = image.convert('L')  # Use 'L' for grayscale

# Extract pixel values
pixel_values = np.array(image)

# Save the pixel values to a file
np.save('pixel_values.npy', pixel_values)

# Load the pixel values from the saved file
loaded_pixel_values = np.load('pixel_values.npy')

# Create an RGBA image based on the loaded pixel values
height, width = loaded_pixel_values.shape
glow_image = np.zeros((height, width, 4), dtype=np.uint8)  # 4 channels for RGBA

# Define characters for display
characters = "HAPPY TIHAR LAXMI PUJA 2081"

# Define the size of each character block
char_block_size = 2  # Adjust this value to increase/decrease character size

# Create a figure and axis for the animation
fig, ax = plt.subplots(figsize=(10, 5))
image_display = ax.imshow(glow_image, animated=True)
ax.axis('off')  # Turn off axis

# Function to generate a random bright color
def random_real_bright_color():
    hue = random.uniform(0, 1)
    saturation = random.uniform(0.7, 1)
    lightness = random.uniform(0.5, 0.8)
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    return [int(255 * c) for c in rgb] + [255]  # Full opacity

# Function to generate a random dim color
def random_real_dim_color():
    hue = random.uniform(0, 1)
    saturation = random.uniform(0.2, 0.5)
    lightness = random.uniform(0.1, 0.3)
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    return [int(255 * c) for c in rgb] + [255]  # Full opacity

# Pre-generate unique color pairs for each character
character_color_map = {char: (random_real_bright_color(), random_real_dim_color()) for char in set(characters)}

# Generate background data points with unique color pairs
num_background_points = 100  # Number of background points
point_radius = 2  # Radius of each background point
background_points = []
for _ in range(num_background_points):
    x = random.randint(point_radius, width - point_radius)
    y = random.randint(point_radius, height - point_radius)
    on_color = random_real_bright_color()
    off_color = random_real_dim_color()
    background_points.append((x, y, on_color, off_color))

# Animation parameters
frame_offset = 0  # To control the movement direction
scroll_speed = 1  # Speed of scrolling

# Function to draw a circle on the glow image
def draw_circle(image, center, radius, color):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    cx, cy = center
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if mask[dy + radius, dx + radius]:
                if 0 <= cy + dy < image.shape[0] and 0 <= cx + dx < image.shape[1]:
                    image[cy + dy, cx + dx] = color

# Animation update function with pre-generated color combinations
def update(frame):
    global frame_offset
    # Clear glow image for each frame
    glow_image[:, :] = 0  # Reset to transparent

    # Update the frame offset for scrolling effect
    frame_offset += scroll_speed
    frame_offset %= len(characters)

    # Update the characters in the main image
    for i in range(height):
        for j in range(width):
            if loaded_pixel_values[i, j] < 128:  # Assuming dark pixels are letter pixels
                char_index = (j // char_block_size + frame_offset) % len(characters)
                char = characters[char_index]
                on_color, off_color = character_color_map[char]

                intensity = (math.sin(frame * 0.1 + char_index) + 1) / 2
                current_color = [int(on_color[k] * intensity + off_color[k] * (1 - intensity)) for k in range(4)]

                for k in range(char_block_size):
                    for l in range(char_block_size):
                        if i + k < height and j + l < width:
                            glow_image[i + k, j + l] = current_color

    # Update the background data points with pulsing effect
    for (x, y, on_color, off_color) in background_points:
        # Calculate the pulsing intensity for each point independently
        intensity = (math.sin(frame * 0.1 + x + y) + 1) / 2
        current_color = [int(on_color[k] * intensity + off_color[k] * (1 - intensity)) for k in range(4)]

        # Draw the background point as a circle
        draw_circle(glow_image, (x, y), point_radius, current_color)

    # Update the display image
    image_display.set_array(glow_image)

    return image_display,

# Create an animation with a faster interval
ani = FuncAnimation(fig, update, frames=100, interval=1, blit=True)  # Decreased interval for faster animation

# Show the animation
plt.show()
