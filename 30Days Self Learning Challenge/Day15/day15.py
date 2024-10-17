import cv2
import os
import requests
import numpy as np

def load_image(file_path, url=None):
    # Check if the image exists locally or should be downloaded
    if os.path.exists(file_path) and file_path != '':
        print(f"Loading image from local file system: {file_path}")
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    elif url:
        print(f"Local file not found. Downloading image from URL: {url}")
        # Download the image from the URL
        response = requests.get(url)
        image_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    else:
        print(f"Image not found locally or at URL: {file_path}")
        return None
    return image

def process_images(image_paths, urls=None):
    # Ensure we have at least as many URLs as paths; extend image_paths if needed
    if urls and len(urls) > len(image_paths):
        # Fill missing paths with empty strings if more URLs are provided
        image_paths.extend([''] * (len(urls) - len(image_paths)))

    for i in range(len(image_paths)):
        image_path = image_paths[i]
        url = urls[i] if i < len(urls) else None

        # Load the image either from the local path or the URL
        image = load_image(image_path, url)
        
        if image is None:
            continue
        
        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred_image, 100, 200)

        # Concatenate the original and edge-detected images horizontally
        concatenated_images = np.hstack((image, edges))

        # Display the side-by-side images
        cv2.imshow(f'Original and Edges for Image {i+1}', concatenated_images)

        # Wait for a key press, then close the current image before proceeding
        print(f"Displaying Image {i+1}. Press any key to close and proceed to the next image.")
        cv2.waitKey(0)  # 0 means wait indefinitely for a key press
        cv2.destroyAllWindows()  # Close the current image before showing the next

# List of local image paths (can be shorter than URL list or empty)
image_paths = ['','','',]  # Add local image paths if available

# Corresponding list of URLs if the images are not found locally
image_urls = [
    'https://www.melbourneradiology.com.au/wp-content/uploads/2021/06/MRI-BRAIN-WITH-CONTRAST-0001.jpg',
    'https://www.melbourneradiology.com.au/wp-content/uploads/2021/06/MRI-BRAIN-WITH-CONTRAST-0003.jpg',
    'https://prod-images-static.radiopaedia.org/images/50175602/8474b9eda07dbabbfb972770e3fa67_big_gallery.jpeg',
    'https://www.ganeshdiagnostic.com/admin/public/assets/images/product/1665383755-mri%20kidneys.webp',
    'https://dt5vp8kor0orz.cloudfront.net/5d94bca9a246fa969c923e609e77cadb9106e5e4/3-Figure2-1.png'
]

# Process and display multiple images, one at a time
process_images(image_paths, image_urls)


"""
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
(5, 5): This tuple defines the size of the Gaussian kernel. The values (5, 5) indicate that the kernel will be a 5x5 matrix. 
A larger kernel results in a more significant blur effect. The kernel size must be positive and odd (e.g., 1, 3, 5, 7, ...).
0: This value represents the standard deviation in the X and Y directions. A value of 0 means that the standard deviation 
is calculated based on the kernel size. If you specify a positive number, it will be used directly to control the amount of blur.

"""


"""
edges = cv2.Canny(blurred_image, 100, 200)
Parameters:
blurred_image: The input image (blurred) on which you want to perform edge detection.
100: This is the lower threshold for edge detection. It determines the minimum intensity gradient required to consider a pixel as an edge.
Pixels with gradient values below this threshold are discarded.
200: This is the upper threshold for edge detection. 
It indicates the maximum intensity gradient for a pixel to be considered as part of an edge. 
Pixels with gradient values above this threshold are marked as strong edges. Pixels with gradient values between the two thresholds are 
considered weak edges and may be retained depending on their connectivity to strong edges.

Summary:
Gaussian Blur is used to reduce noise and detail in the image, which helps improve the performance of edge detection.
Canny Edge Detection identifies edges in the image based on intensity gradients, using the two specified thresholds to control which pixels are 
considered edges.

"""