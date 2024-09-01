import numpy as np
from PIL import Image
import math
import cv2

from PIL import Image

# Load the image
image_path = "output_image.png" # Change this to the path of the image you want to convert
img = Image.open(image_path)

# Convert the image to grayscale
img = img.convert("L")

# Get the image dimensions
width, height = img.size

# Extract pixel data
pixels = list(img.getdata())

# Convert the pixel data into a string format
pixel_str = ""
i = 0
for y in range(height):
    for x in range(width):
        i+=1
        grayscale_value = pixels[y * width + x]
        pixel_str += f"{grayscale_value} "

print(i)
print(pixel_str)