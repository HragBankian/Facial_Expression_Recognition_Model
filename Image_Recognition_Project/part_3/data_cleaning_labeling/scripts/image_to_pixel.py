import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

img = cv2.imread(f'1.jpg')

resized_img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
gray_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

# save the gray_image as a png file
cv2.imwrite(f'modified.png', gray_image)



# Load the image
image_path = "modified.png" # Change this to the path of the image you want to convert

img = Image.open(image_path)


# Convert the image to grayscale
img = img.convert("L")

# Convert 48 by 48 image 
img = img.resize((48, 48), Image.Resampling.LANCZOS)


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