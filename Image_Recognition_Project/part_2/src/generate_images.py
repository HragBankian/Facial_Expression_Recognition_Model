import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# read data.csv and get 25 focused images save them as .png in generate_images folder

df = pd.read_csv('../data/data.csv')
focused = df[df['emotion'] == 1]

for i in range(25):
    img = focused.iloc[i]['pixels']
    img = np.array([int(p) for p in img.split()]).reshape(48, 48)
    plt.imsave(f'generate_images/{i}.png', img, cmap='gray')


