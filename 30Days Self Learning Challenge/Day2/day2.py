import requests
from PIL import Image
import numpy as np
from io import BytesIO

# Function to convert image from URL to vector and save to .txt file
def image_url_to_vector(image_url, output_file):
    # Fetch the image from the URL
    response = requests.get(image_url)
    
    # Open the image from the response content
    image = Image.open(BytesIO(response.content))
    
    # Convert the image to RGB (if not already in that format)
    image = image.convert("RGB")
    
    # Resize the image to a fixed size (optional)
    image = image.resize((100, 100))  # Resize to 100x100 pixels
    
    # Convert the image to a NumPy array
    image_array = np.array(image)
    
    # Flatten the array to create a vector
    image_vector = image_array.flatten()

    # Save the 1D vector, 2D matrix, and 3D matrix to a .txt file
    with open(output_file, 'w') as f:
        # Write 2D matrix representation
        f.write("2D Matrix Representation:\n")
        for row in image_array:
            f.write(' '.join(map(str, row)) + '\n')
        
        # Write 1D vector representation
        f.write("\n1D Vector Representation:\n")
        f.write(' '.join(map(str, image_vector)) + '\n')
        
        # Write 3D matrix representation (each RGB channel)
        f.write("\n3D Matrix Representation (RGB Channels):\n")
        for i in range(image_array.shape[0]):  # Iterate over rows
            for j in range(image_array.shape[1]):  # Iterate over columns
                # Write the RGB values for each pixel
                r, g, b = image_array[i, j]
                f.write(f'[{i}, {j}] -> R: {r}, G: {g}, B: {b}\n')

    return image_vector

# Example usage
image_url = 'https://unsplash.it/100'  # Specify your image URL
output_file = 'image_vector.txt'  # Output file name
vector = image_url_to_vector(image_url, output_file)

# Print the shape of the vector
print("Vector shape:", vector.shape)
print("Vector representation saved to:", output_file)
