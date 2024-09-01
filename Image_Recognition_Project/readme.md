# Comp 472 Image Recognition Project - SMART A.I.ssistant
- By Hrag Bankian, Gevorg Markarov, and Jaspreet Singh

# Overview
This project aims to detect student sentiment in a classroom by using CNN to recognize the following sentiments:
- Angry
- Happy
- Focused
- Neutral
# Data Set
We found a prelabeled databset that already contains labeled grayscale 48x48 images for Angry, Happy, and Neutral. The dataset we got was in two forms:
- Labeled images. The data set link can be found [here](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset/data). In terms of explanation of the dataset, there is nothing available with it.

- CSV file. The data set link can be found [here](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data)

# Files
## Data Visualization
This folder contains
- pid: It contains pixel intensity distribution histograms for each class of emotions.
- sample_images: It contains grid of sample images along with their histograms for each class of emotions.
- class_distribution.png: It is a bar graph showing the number of images in each class of emotions
- data_visualization.py: It is a script that creates pixel intensity distributions for each class of emotions, samples 15 images and creates histograms for them and places them in a grid, and finally it creates a bar graph of the number of images in the dataset for each emotion class. To run the script, please make sure the path to the csv file containing the dataset is correct. The dataset file is complete_fer2013.csv and currently it is in the part_1/data_cleaning_labeling/data/data_v3_complete/ folder. It is likely that the dataset is zipped as data_v3_compelete.zip file in ata_cleaning_labeling/data/ folder. Simply unzip it in that folder. This file contains the following functions
  - ImagesDistribution(df: pd.DataFrame) : This function creates a bar graph for the number of images in each class of emotions
  - PixelIntensityDistribution(df: pd.DataFrame) : This function creates histograms for the pixel intensity distribution for all images in each class of emotions
  - SampleImages(df: pd.DataFrame, nbOfSamples: int, show: bool) : This function samples 'nbOfSamples' images from the dataset for each class and saves those images and then pixel intensity distribution histograms
  - SampleImagesGrid(df: pd.DataFrame, nbOfSamples: int, show: bool = False) : This function does exactly the same as SampleImages but rather than saving the images and their histograms individually it creates a grid for each emotion and saves it.
 
## data_cleaning_labeling

- data: contains the dataset as csv files. There are different versions of that file, according to the modification that we made previously. (csv files are compressed, we uncompress it when we put it on our machines. It's done to avoid github's 100mb file limiter) 
- personal_images: contains the images of the group members that we later put into the dataset
- save_focused_count: text files to keep track of indexes in the script files
- scripts: all the python code used to polish the dataset

## fer2013_dataset:
- contains all the dataset that we took online in a raw form.


## doing an example change to check lfs