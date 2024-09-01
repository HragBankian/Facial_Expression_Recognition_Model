import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

import time
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_data(df: pd.DataFrame):
    images = np.array([np.fromstring(image, sep=' ').reshape(48, 48) for image in df['pixels']])
    images = images / 255.0  # Normalize pixel values
    labels = df['emotion'].values
    return images, labels

# Preprocess data
def create_dataloader(train_data: pd.DataFrame, test_data: pd.DataFrame, validation_data: pd.DataFrame) -> tuple[DataLoader, DataLoader, DataLoader]:
    # Load datasets
    train_images, train_labels = preprocess_data(train_data)
    test_images, test_labels = preprocess_data(test_data)
    validation_images, validation_labels = preprocess_data(validation_data)


    # Convert numpy arrays to PyTorch tensors
    train_images = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_images = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    validation_images = torch.tensor(validation_images, dtype=torch.float32).unsqueeze(1)
    validation_labels = torch.tensor(validation_labels, dtype=torch.long)
    print("Converted numpy arrays to PyTorch tensors")

    # Create PyTorch datasets
    # TensorDataset  is a Dataset wrapping tensors. Each sample will be retrieved by indexing tensors along the first dimension.
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    validation_dataset = TensorDataset(validation_images, validation_labels)
    print("Created PyTorch datasets")
    # Create data loaders 
    # Dataloader is an optimized data loading utility class that helps in efficient data loading and shuffling.
    return DataLoader(train_dataset, batch_size=64, shuffle=True), DataLoader(test_dataset, batch_size=64, shuffle=False), DataLoader(validation_dataset, batch_size=64, shuffle=False)
    



# Define the CNN model
class EmotionCNN(nn.Module):
    def __init__(self, kernel_size: int = 7, layers=3):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=kernel_size, padding=int(kernel_size / 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=kernel_size, padding=int(kernel_size / 2))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=int(kernel_size / 2))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 4)  # 4 classes: angry, focused, neutral, happy
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define the CNN model
class EmotionCNNKLayers(nn.Module):
    def __init__(self, kernel_size: int = 3, layers=3):
        super(EmotionCNNKLayers, self).__init__()
        self.conv_layers = []
        first = 1
        second = 32
        for i in range(layers):
            self.conv_layers.append(nn.Conv2d(first, second, kernel_size=kernel_size, padding=int(kernel_size / 2)))
            first = second
            second *=  2
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.pool = nn.MaxPool2d(2, 2)
        end_dimension = int(48 / math.pow(2, layers))
        self.layers = layers
        self.kernel_size = kernel_size
        self.final_pool_dimension = first * end_dimension * end_dimension
        self.neurons = 512
        self.fc1 = nn.Linear(self.final_pool_dimension, self.neurons)
        self.fc2 = nn.Linear(self.neurons, 4)  # 4 classes: angry, focused, neutral, happy
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        for i in range(self.layers):
            x = self.pool(nn.ReLU()(self.conv_layers[i](x)))
        x = x.view(-1, self.final_pool_dimension)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



def train(train_loader: DataLoader, validation_loader: DataLoader, kernel_size: int = 3, layers: int = 3, epochs: int = 10, patience_: int = 3, lr= 0.001, save_path= '../models/best_model_test_k') -> nn.modules:
    '''
        This function trains the given model for the given number of epochs, patience, and lr.
        Epochs is the number of times we train our model on the training dataset and then test
        it against the validation set. If the validation loss is greater than the training loss
        for patience_ number of times in a row, we end the training to avoid overfitting. We also
        save the best after each epoch.
        
        Parameters
        ----------
            model : EmotionCNN
                The model that we are training.
            epochs : int = 10
                The number of times we loop over our training dataset, and test the model against
                the validation set
            patience_ : int = 3
                The consecutive number of times our validation loss is greater than the training loss
                after which we stop training.
            lr : int = 0.001
                The learning rate used for training.
    '''
    
    model = EmotionCNNKLayers(kernel_size=kernel_size, layers=layers).to(device=device)
    # Define loss function and optimizer
    
    #if layers == 3:
    weight = torch.tensor([10.0,30.0,7.0,5.0]).to(device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    # else:
    #     criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    print("Defined loss function and optimizer")

    # Train the model
    num_epochs = epochs
    patience = patience_
    best_loss = float('inf')
    # read best_loss from models/best_loss.txt
    # try:
    #     with open('part_2/models/best_loss.txt', 'r') as f:
    #         best_loss = float(f.read())
    # except FileNotFoundError:
    #     pass
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device=device), labels.to(device=device)
            optimizer.zero_grad()
            torch.cuda.synchronize()
            outputs = model(inputs)
            torch.cuda.synchronize()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        validation_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device=device), labels.to(device=device)
                torch.cuda.synchronize()
                outputs = model(inputs)
                torch.cuda.synchronize()
                loss = criterion(outputs, labels)
                validation_loss += loss.item()

        print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {validation_loss/len(validation_loader)}")

        # Early stopping
        if validation_loss < best_loss:
            best_loss = validation_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path + str(kernel_size) + "_l" + str(layers) + '_n' + str(model.neurons) + '.pth')  # Save the best model
            #best_loss store in models/best_loss.txt
            # with open('part_2/models/best_loss.txt', 'w') as f:
            #     f.write(str(best_loss))
            # print("Save the best model")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break
    print("Trained and validated the model")
    return model

# Load the best model
def load_saved_model(model: nn.Module, test_loader: DataLoader, path='../models/best_model_test_k') -> None:
    kernel_size = model.kernel_size
    layers = model.layers
    model.load_state_dict(torch.load(path + str(kernel_size) + "_l" + str(layers) + '_n' + str(model.neurons) + '.pth'))

    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    true_labels = []
    output_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            true_labels.extend(labels.tolist())
            output_labels.extend(predicted.tolist())

    print(f"Test Loss: {test_loss/len(test_loader)}, Test Accuracy: {100 * correct / total}%")
    return true_labels, output_labels
   

def score_image(model: nn.Module, img: DataLoader) -> int:
    
    with torch.no_grad():
        for inputs, labels in img:
            inputs = inputs.to(device)
            output = model(inputs)
            _, predicted = torch.max(output, 1)
            
            return int(predicted)
        
        
        
        