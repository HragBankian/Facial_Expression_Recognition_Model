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
df = pd.read_csv('../data/data_complete.csv')
# read the last image you checked
startingPoint = 0
try:
    with open('last_image.txt', 'r') as file:
        startingPoint = int(file.read())
except FileNotFoundError:
    pass
def end():
    global newdf
    global startingPoint
    global i
    
    #read label_data.csv and append the new data to it

    try:
        olddf = pd.read_csv(f'labeled_data.csv')
        newdf = pd.concat([olddf, newdf], ignore_index=True)
    except FileNotFoundError:
        pass

    # storing i into last_image.txt
    with open('last_image.txt', 'w') as file:
        file.write(str(i))
    # Save the updated DataFrame to a new CSV file
    newcsv = f'labeled_data.csv'
    newdf.to_csv(newcsv, index=False)
    
    print(f"Updated data saved to '{newcsv}'")
    
    # display a table showing how many of each gender and age group are in the new data
    

    exit()
# Print a table containing the number of each gender an age group
some = pd.read_csv(f'labeled_data.csv')

# Count for each emotion
count = {'y-m': {0: 0, 1: 0, 2: 0, 3: 0}, 'y-f': {0: 0, 1: 0, 2: 0, 3: 0}, 'y-o': {0: 0, 1: 0, 2: 0, 3: 0},
         'm-m': {0: 0, 1: 0, 2: 0, 3: 0}, 'm-f': {0: 0, 1: 0, 2: 0, 3: 0}, 'm-o': {0: 0, 1: 0, 2: 0, 3: 0},
         'o-m': {0: 0, 1: 0, 2: 0, 3: 0}, 'o-f': {0: 0, 1: 0, 2: 0, 3: 0}, 'o-o': {0: 0, 1: 0, 2: 0, 3: 0}}

for index, row in some.iterrows():
    count[f"{row['age']}-{row['gender']}"][int(row['emotion'])] += 1

# Table of counts
print(f"""Counts
+--------+---+---+---+---+
|        | 0 | 1 | 2 | 3 |
+--------+---+---+---+---+
| Y-M    |{count['y-m'][0]:^3}|{count['y-m'][1]:^3}|{count['y-m'][2]:^3}|{count['y-m'][3]:^3}|
| Y-F    |{count['y-f'][0]:^3}|{count['y-f'][1]:^3}|{count['y-f'][2]:^3}|{count['y-f'][3]:^3}|
| Y-O    |{count['y-o'][0]:^3}|{count['y-o'][1]:^3}|{count['y-o'][2]:^3}|{count['y-o'][3]:^3}|
| M-M    |{count['m-m'][0]:^3}|{count['m-m'][1]:^3}|{count['m-m'][2]:^3}|{count['m-m'][3]:^3}|
| M-F    |{count['m-f'][0]:^3}|{count['m-f'][1]:^3}|{count['m-f'][2]:^3}|{count['m-f'][3]:^3}|
| M-O    |{count['m-o'][0]:^3}|{count['m-o'][1]:^3}|{count['m-o'][2]:^3}|{count['m-o'][3]:^3}|
| O-M    |{count['o-m'][0]:^3}|{count['o-m'][1]:^3}|{count['o-m'][2]:^3}|{count['o-m'][3]:^3}|
| O-F    |{count['o-f'][0]:^3}|{count['o-f'][1]:^3}|{count['o-f'][2]:^3}|{count['o-f'][3]:^3}|
| O-O    |{count['o-o'][0]:^3}|{count['o-o'][1]:^3}|{count['o-o'][2]:^3}|{count['o-o'][3]:^3}|
+--------+---+---+---+---+
| Y-SUM  | {sum(count['y-m'].values()) +sum(count['y-f'].values()) +sum(count['y-o'].values()):^3}
| M-SUM  | {sum(count['m-m'].values()) +sum(count['m-f'].values()) +sum(count['m-o'].values()):^3}
| O-SUM  | {sum(count['o-m'].values()) +sum(count['o-f'].values()) +sum(count['o-o'].values()):^3}
+--------+---+---+---+---+
| M-SUM  | {sum(count['y-m'].values()) +sum(count['m-m'].values()) +sum(count['o-m'].values()):^3}
| F-SUM  | {sum(count['y-f'].values()) +sum(count['m-f'].values()) +sum(count['o-f'].values()):^3}
| O-SUM  | {sum(count['y-o'].values()) +sum(count['m-o'].values()) +sum(count['o-o'].values()):^3}
+--------+---+---+---+---+
| Total  | {some.shape[0]:^3}
      """)
# Put the table into age_gender_table.txt


i = 0
# Create a new DataFrame to store the
newdf = pd.DataFrame(columns=['emotion', 'pixels' ,'age', 'gender'])
# We need to add labels to the neutral images. We need to specify their gender (Male, Female, Others) and their age (Young, Middle, Old) 50 samples at a time
for index, row in df.iterrows():
    if i < startingPoint:
        i += 1
        continue

    # Display the image
    display_image(row['pixels'])
    
    # Ask the user for the label
    age = None
    print("Please enter the age of the person: k = young, l = middle, ; = old")
    while True:
        age = input()
        if age == '1':
            end()
        if age == 'k' or age == 'l' or age == ';':
            
            if age == 'k':
                age = 'y'
            elif age == 'l':
                age = 'm'
            elif age == ';':
                age = 'o'
            break
        print("Invalid input. Please enter k, l, or ;.")
    
    print("Please enter the gender of the person: k = male, l = female, ; = other")
    gender = None
    while True:
        gender = input()
        if gender == '1':
            end()
        if gender == 'k' or gender == 'l' or gender == ';':
            if gender == 'k':
                gender = 'm'
            elif gender == 'l':
                gender = 'f'
            elif gender == ';':
                gender = 'o'
            break
        print("Invalid input. Please enter k, l, or ;.")
    
    # Update the data frame
    newdf = newdf._append({'emotion': row['emotion'], 'pixels': row['pixels'], 'age': age, 'gender': gender}, ignore_index=True)


    i += 1


