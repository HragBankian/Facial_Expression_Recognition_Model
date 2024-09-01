# import cv2
# import os
from deepface import DeepFace
import json
import pandas as pd
import deepface
image_paths = []

for i in range(0,20000):
    image_paths.append(f'generate_images/{i}.png')

length = len(image_paths)
print(length)
data = pd.read_csv('data_complete_updated.csv')


for i, row in data.iterrows():
    
    if i % 1000 == 0 :
        print(i)
        data.to_csv('data_complete_updated2.csv', index=False)
    try: 
        if i  >= length:
            break
        result = DeepFace.analyze(image_paths[i], actions=['gender'], enforce_detection=False)
    # Analyze the image for age and gender
        result_json = json.dumps(result)

        # Parse the JSON string back into a Python object (list of dictionaries)
        result_dict = json.loads(result_json)

        data.at[i, 'gender'] = 'm' if result_dict[0]['dominant_gender'] == 'Man' else 'f'
        

    except Exception as e:
        print(f"Error processing image {i}: {e}")
# Save the updated dataframe to CSV file
data.to_csv('data_complete_updated2.csv', index=False)

print("Updated dataframe saved to data_complete_update2.csv")






