import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import zipfile
import os
import pathlib

# Define custom dataset
class EmotionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data.iloc[idx, 1]
        img = [int(p) for p in img.split()]
        img = torch.tensor(img, dtype=torch.float32).view(48, 48)  # Assuming images are 48x48 pixels
        label = self.data.iloc[idx, 0]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label


# Define file paths
top_level_dir = pathlib.Path(os.getcwd()).parent.parent.absolute().as_posix()
data_dir_path = top_level_dir + "/" + "part_2/data/"
complete_path = top_level_dir + "/" + "part_2/data/balanced_data_complete.csv"

train_path = top_level_dir + "/" + "part_2/data/balanced_data_train.csv"
train_path_zip = top_level_dir + "/" + "part_2/data/balanced_data_train.zip"

validation_path = top_level_dir + "/" + "part_2/data/balanced_data_validation.csv"
validation_path_zip = top_level_dir + "/" + "part_2/data/labeled_data_validation.zip"

test_path = top_level_dir + "/" + "part_2/data/balanced_data_test.csv"
test_path_zip = top_level_dir + "/" + "part_2/data/labeled_data_test.zip"


# Split the dataset
def split_dataset(train_size: float = 0.7, test_size: float = 0.15) -> None:
    '''
        This function reads the "data_complete.csv" file and splits the data into 3 parts:
            1. train : By default it is 70% of the complete data. It is stored in "part_2/data/data_train.csv"
            2. validation : By default it is 15% of the complete data. It is stored in "part_2/data/data_validation.csv"
            3. test : By default it is 15% of the compelete data. It is stored in "part_2/data/data_train.csv"
            
        Parameters
        ----------
            train_size : float
                This should be a float ∈ [0, 1] which signifies the percentage of the total dataset that should be used for training
            
            test_size : float
                This should be a float ∈ [0, 1 - train_size] which signifies the percentage of total data that should be used for
                testing. Any remaining data is assumed to be used for validation.
            
        Returns
        -------
            None
    '''
    # Load the dataset
    data = pd.read_csv(complete_path)
    train_data, test_data = train_test_split(data, test_size=(1 - train_size), random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=((1 - train_size) / 2), random_state=42)
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(validation_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    
def get_dataset(train_path = train_path, test_path = test_path, validation_path = validation_path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
        This function reads csv files containing the training, testing, and validation datasets 
        and returns them as a pd.DataFrame. The paths for the files are given above. If the files do 
        not exist, then it extracts the zip files containing the csv files and then reads them.
        This function must be run in the "part_2/src/" directory
        
        Parameters
        ----------
            None
            
        Returns
        -------
        
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] : A tuple of dataframes containing training, testing, and validation
                datasets in the aforementioned order.
            
    '''
    

    # check whether the csv file exists
    if not os.path.exists(train_path):
        # if not then extract the file
        with zipfile.ZipFile(train_path_zip, 'r') as file:
            file.extractall(data_dir_path)
    
    if not os.path.exists(test_path):
        with zipfile.ZipFile(test_path_zip, 'r') as file:
            file.extractall(data_dir_path)
        
    if not os.path.exists(validation_path):
        with zipfile.ZipFile(validation_path_zip, 'r') as file:
            file.extractall(data_dir_path)
    
    return pd.read_csv(train_path),  pd.read_csv(test_path), pd.read_csv(validation_path)
        


