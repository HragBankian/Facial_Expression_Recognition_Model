import numpy as np
import pandas as pd
from PIL import Image

# Read the CSV file
df = pd.read_csv('data_cleaning_labeling/data/data_complete.csv')

# Iterate over each row in the dataframe
for i in range(len(df)):
    try:
        # Get the pixel string from the dataframe and convert it to a numpy array
        img_pixels = df.iloc[i]['pixels']
        img = np.array([int(p) for p in img_pixels.split()], dtype=np.uint8).reshape(48, 48)
        
        # Create an Image object from the numpy array
        img = Image.fromarray(img, mode='L')  # mode='L' for grayscale
        
        # Save the image as PNG in the generate_images folder
        img.save(f'generate_images/{i}.png')
        
        print(f"Image {i} saved successfully.")
    except Exception as e:
        print(f"Error processing image {i}: {e}")