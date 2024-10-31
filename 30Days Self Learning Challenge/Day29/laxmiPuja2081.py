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
    hue = random.uniform(0, 1)  # Hue between 0 and 1 (all colors)
    saturation = random.uniform(0.7, 1)  # High saturation for vibrancy
    lightness = random.uniform(0.5, 0.8)  # Medium lightness for brightness
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)  # Convert HSL to RGB
    return [int(255 * c) for c in rgb] + [255]  # Convert to 0-255 scale and add full opacity

# Function to generate a random dim color (for off state)
def random_real_dim_color():
    hue = random.uniform(0, 1)
    saturation = random.uniform(0.2, 0.5)  # Lower saturation for a dim effect
    lightness = random.uniform(0.1, 0.3)  # Low lightness for dimness
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    return [int(255 * c) for c in rgb] + [random.randint(0, 128)]  # Add semi-transparency

# Pre-generate unique color pairs for each character
character_color_map = {char: (random_real_bright_color(), random_real_dim_color()) for char in set(characters)}

# Animation parameters
frame_offset = 0  # To control the movement direction
scroll_speed = 1  # Speed of scrolling

# Animation update function with pre-generated color combinations
def update(frame):
    global frame_offset
    # Clear glow image for each frame
    glow_image[:, :] = 0  # Reset to transparent

    # Update the frame offset for scrolling effect
    frame_offset += scroll_speed

    # Keep frame_offset within bounds
    frame_offset %= len(characters)

    for i in range(height):
        for j in range(width):
            if loaded_pixel_values[i, j] < 128:  # Assuming dark pixels are letter pixels
                # Calculate char_index using frame_offset
                char_index = (j // char_block_size + frame_offset) % len(characters)
                
                # Get the character and its color pair
                char = characters[char_index]
                on_color, off_color = character_color_map[char]

                # Calculate intensity based on frame to create a pulsing effect
                intensity = (math.sin(frame * 0.1 + char_index) + 1) / 2  # Varies between 0 and 1
                current_color = [int(on_color[k] * intensity + off_color[k] * (1 - intensity)) for k in range(4)]

                # Draw the glow with pre-generated colors
                for k in range(char_block_size):  # Fill a block for the character
                    for l in range(char_block_size):
                        if i + k < height and j + l < width:  # Ensure within bounds
                            glow_image[i + k, j + l] = current_color

    # Update the display image
    image_display.set_array(glow_image)

    return image_display,

# Create an animation
ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

# Show the animation
plt.show()
