from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

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

# Define characters and their corresponding bright colors
characters = "HAPPY TIHAR LAXMI PUJA 2080"
on_colors = [
    [0, 255, 0, 255],    # H - Bright green (RGBA)
    [0, 0, 255, 255],    # A - Bright blue (RGBA)
    [255, 0, 0, 255],    # P - Bright red (RGBA)
    [255, 0, 0, 255],    # P - Bright red (RGBA)
    [255, 165, 0, 255],  # Y - Bright orange (RGBA)
    [75, 0, 130, 255],   # T - Bright indigo (RGBA)
    [255, 20, 147, 255], # I - Bright deep pink (RGBA)
    [0, 255, 255, 255],  # R - Bright cyan (RGBA)
    [128, 0, 128, 255],  # L - Bright purple (RGBA)
    [0, 0, 255, 255],    # A - Bright blue (RGBA)
    [0, 0, 255, 255],    # X - Bright blue (RGBA)
    [255, 0, 0, 255],    # M - Bright red (RGBA)
    [255, 20, 147, 255], # I - Bright deep pink (RGBA)
    [255, 0, 0, 255],    # P - Bright red (RGBA)
    [255, 20, 147, 255], # U - Bright deep pink (RGBA)
    [0, 255, 255, 255],  # J - Bright cyan (RGBA)
    [0, 255, 0, 255],    # A - Bright green (RGBA)
    [255, 255, 0, 255],  # 2 - Bright yellow (RGBA)
    [0, 0, 0, 0],        # 0 - Fully transparent
    [255, 165, 0, 255],  # 8 - Bright orange (RGBA)
    [0, 0, 0, 0]         # 0 - Fully transparent
]

# Define off colors corresponding to each character (using lighter shades)
off_colors = [
    [255, 255, 255, 0],  # H - Fully transparent
    [240, 248, 255, 0],  # A - Fully transparent
    [255, 240, 245, 0],  # P - Fully transparent
    [255, 240, 245, 0],  # P - Fully transparent
    [255, 255, 224, 0],  # Y - Fully transparent
    [230, 230, 250, 0],  # T - Fully transparent
    [255, 240, 245, 0],  # I - Fully transparent
    [240, 255, 255, 0],  # R - Fully transparent
    [230, 230, 250, 0],  # L - Fully transparent
    [240, 248, 255, 0],  # A - Fully transparent
    [240, 248, 255, 0],  # X - Fully transparent
    [255, 228, 225, 0],  # M - Fully transparent
    [255, 240, 245, 0],  # I - Fully transparent
    [255, 228, 225, 0],  # P - Fully transparent
    [255, 240, 245, 0],  # U - Fully transparent
    [240, 248, 255, 0],  # J - Fully transparent
    [240, 248, 255, 0],  # A - Fully transparent
    [255, 255, 224, 0],  # 2 - Fully transparent
    [255, 255, 255, 0],  # 0 - Fully transparent
    [255, 255, 224, 0],  # 8 - Fully transparent
    [255, 255, 255, 0]   # 0 - Fully transparent
]

# Define the size of each character block
char_block_size = 2  # Adjust this value to increase/decrease character size

# Create a figure and axis for the animation
fig, ax = plt.subplots(figsize=(10, 5))
image_display = ax.imshow(glow_image, animated=True)
ax.axis('off')  # Turn off axis

# Animation parameters
frame_offset = 0  # To control the movement direction
scroll_speed = 1  # Speed of scrolling
animation_interval = random.randint(1, 100)  # Initial random interval

# Animation update function
def update(frame):
    global frame_offset, animation_interval
    # Clear glow image for each frame
    glow_image[:, :] = 0  # Reset to transparent

    # Update the frame offset for scrolling effect
    if frame % 20 < 10:  # Move forward
        frame_offset += scroll_speed
    else:  # Move backward
        frame_offset -= scroll_speed

    # Keep frame_offset within bounds
    frame_offset %= len(characters)

    for i in range(height):
        for j in range(width):
            if loaded_pixel_values[i, j] < 128:  # Assuming dark pixels are letter pixels
                # Calculate char_index using frame_offset
                char_index = (j // char_block_size + frame_offset) % len(characters)

                # Ensure the char_index is within the bounds of on_colors
                if char_index < len(on_colors):
                    # Get the color for the letter
                    on_color = on_colors[char_index]
                    off_color = off_colors[char_index]  # Get the corresponding off color

                    # Slow down the alternation between on and off colors
                    for k in range(char_block_size):  # Fill a block for the character
                        for l in range(char_block_size):
                            if i + k < height and j + l < width:  # Ensure within bounds
                                glow_image[i + k, j + l] = on_color if (frame % 20) < 10 else off_color

    # Update the display image
    image_display.set_array(glow_image)

    # Randomly change the interval for the next frame
    animation_interval = random.randint(1, 100)

    return image_display,

# Create an animation with a random interval
ani = FuncAnimation(fig, update, frames=100, interval=animation_interval, blit=True)

# Show the animation
plt.show()