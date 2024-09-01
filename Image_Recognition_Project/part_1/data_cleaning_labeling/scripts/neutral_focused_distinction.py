import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def display_image(pixel_values, size=48):
    # Convert the pixel string to an array
    pixels = np.array([int(p) for p in pixel_values.split()]).reshape(size, size)
    # Display the image
    plt.imshow(pixels, cmap='gray')
    plt.axis('off')
    plt.show()

# Read the CSV file
df = pd.read_csv('../data/data_v2_focused_class.csv')
# read the last image you checked
startingPoint = 0
try:
    with open('../save_focused_count/last_image.txt', 'r') as file:
        startingPoint = int(file.read())
except FileNotFoundError:
    pass
numberOfFocused = 0
try:
    with open('../save_focused_count/focused_num.txt', 'r') as file:
        numberOfFocused = int(file.read())
except FileNotFoundError:
    pass
i = 0

# Iterate over the rows in the DataFrame
for index, row in df.iterrows():
    
    # Check for emotion 0 and display image
    if i < startingPoint:
        i += 1
        continue
    if row['emotion'] == 0 or row['emotion'] == 2:
        display_image(row['pixels'])
        response = input("Is this a focused person? (1 for Yes, to save where left 2, to remove the row 3, to continue press enter or insert any other string): ")
        
        if response == 'l':
            df.at[index, 'emotion'] = 1
            numberOfFocused+=1
        elif response == '1':
            with open('last_image.txt', 'w') as file:
                file.write(str(i))
            with open('focused_num.txt', 'w') as file:
                file.write(str(numberOfFocused))
            break
        elif response == 'k':
            #remove the current row
            df.drop(index, inplace=True)
    i += 1
    
# Save the updated DataFrame to a new CSV file
df.to_csv('updated_filtered_fer2013.csv', index=False)

print("Updated data saved to 'updated_filtered_fer2013.csv'")