
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Load an online image
url = "https://thumbs.dreamstime.com/b/motorcycle-ride-city-streets-dusk-rider-navigates-urban-landscape-towering-skyscrapers-colorful-advertisements-326906216.jpg"  # Replace with the URL of the online image
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# Convert the image to an OpenCV format (BGR)
image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Edge Detection Techniques
# Sobel Edge Detection
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)
sobel_edges = cv2.convertScaleAbs(sobel_edges)

# Canny Edge Detection
canny_edges = cv2.Canny(image, 100, 200)

# Blurring Techniques
# Gaussian Blur
gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

# Box Blur
box_blur = cv2.blur(image, (5, 5))

# Median Blur
median_blur = cv2.medianBlur(image, 5)

# Filtering Techniques
# Low-Pass Filter (Gaussian)
low_pass_filter = cv2.GaussianBlur(image, (5, 5), 0)

# High-Pass Filter
high_pass_filter = cv2.subtract(image, low_pass_filter)

# Slicing Techniques
# Rectangular Slicing
rect_slice = image[50:200, 50:200]  # Adjust the coordinates as needed

# Create a figure to display all the results
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

# Original Image
axes[0, 0].imshow(image)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')

# Sobel Edge Detection
axes[0, 1].imshow(sobel_edges, cmap='gray')
axes[0, 1].set_title("Sobel Edge Detection")
axes[0, 1].axis('off')

# Canny Edge Detection
axes[0, 2].imshow(canny_edges, cmap='gray')
axes[0, 2].set_title("Canny Edge Detection")
axes[0, 2].axis('off')

# Gaussian Blur
axes[1, 0].imshow(gaussian_blur)
axes[1, 0].set_title("Gaussian Blur")
axes[1, 0].axis('off')

# Box Blur
axes[1, 1].imshow(box_blur)
axes[1, 1].set_title("Box Blur")
axes[1, 1].axis('off')

# Median Blur
axes[1, 2].imshow(median_blur)
axes[1, 2].set_title("Median Blur")
axes[1, 2].axis('off')

# Low-Pass Filter
axes[2, 0].imshow(low_pass_filter)
axes[2, 0].set_title("Low-Pass Filter")
axes[2, 0].axis('off')

# High-Pass Filter
axes[2, 1].imshow(high_pass_filter)
axes[2, 1].set_title("High-Pass Filter")
axes[2, 1].axis('off')

# Rectangular Slicing
axes[2, 2].imshow(rect_slice)
axes[2, 2].set_title("Rectangular Slicing")
axes[2, 2].axis('off')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
