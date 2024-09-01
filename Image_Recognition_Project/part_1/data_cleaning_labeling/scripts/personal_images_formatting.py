import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# index = [0, 0, 0, 0]
# for i in range(1, 13):
#     # Read the image
#     img = cv2.imread(f'../personal_images/personal_images_raw/{i}.jpg')

#     resized_img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
#     gray_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
#     if(i % 4 == 1):
#         index[0] += 1
#         cv2.imwrite(f'../personal_images/personal_images_formatted/n_{index[0]}.png', gray_image)
#     if(i % 4 == 2):
#         index[1] += 1
        
#         cv2.imwrite(f'../personal_images/personal_images_formatted/h_{index[1]}.png', gray_image)
#     if(i % 4 == 3):
#         index[2] += 1
        
#         cv2.imwrite(f'../personal_images/personal_images_formatted/a_{index[2]}.png', gray_image)
#     if(i % 4 == 0):
#         index[3] += 1
#         cv2.imwrite(f'../personal_images/personal_images_formatted/f_{index[3]}.png', gray_image)

# adding the images to the csv file with the emotions and training tag for Usage column

# emotions = {0: 'Angry', 1: 'Focused', 2: 'Neutral', 3: 'Happy'}
# usage = {0: 'Training', 1: 'PublicTest', 2: 'PrivateTest'}

# df = pd.DataFrame(columns=['emotion', 'pixels', 'Usage'])

# def add_image(emotion,image_path):
#     global df
#     img = Image.open(image_path)

#     # Convert the image to grayscale
#     img = img.convert("L")

#     # Get the image dimensions
#     width, height = img.size

#     # Extract pixel data
#     pixels = list(img.getdata())

#     # Convert the pixel data into a string format
#     pixel_str = ""
#     i = 0
#     for y in range(height):
#         for x in range(width):
#             i+=1
#             grayscale_value = pixels[y * width + x]
#             pixel_str += f"{grayscale_value} "
        
#     df = df._append({'emotion': emotion, 'pixels': pixel_str, 'Usage': 'Training'}, ignore_index=True)

# for i in range(1, 4):
#     add_image(0,f'../personal_images/personal_images_formatted/a_{i}.png')

# for i in range(1, 4):
#     add_image(1,f'../personal_images/personal_images_formatted/f_{i}.png')
# for i in range(1, 4):
#     add_image(2,f'../personal_images/personal_images_formatted/n_{i}.png')
# for i in range(1, 4):
#     add_image(3,f'../personal_images/personal_images_formatted/h_{i}.png')
    
# df.to_csv('../personal_images/personal_images_formatted.csv', index=False)

df_peronsal = pd.read_csv('../personal_images/personal_images_formatted.csv') # this file does not exist in the repository anymore
df_updated = pd.read_csv('../data/data_v2_focused_class.csv')

pd.concat([df_peronsal, df_updated]).to_csv('../data/data_v3_complete.csv', index=False)