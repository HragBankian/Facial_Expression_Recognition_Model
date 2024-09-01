import data_to_array as dta
import data_split as ds
import time
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
import torch
import math

def plot_confusion_matrix(test_loader:DataLoader,  model: nn.Module = None, path="/part_2/confusion_matrix/conf_mtx_k", model_path="../models/best_model_test_k") -> np.ndarray:
    '''
        This function reads a pretraied model and tests it against the test set. 
        Then it plots the confusion matrix for it and saves it in the 'part_2/confusion_matrix'
        directory with the name of 'conf_mtx_k3_l3" where k means the kernel size of the model
        and l means the number convolutional layers. The user can provide a model to the function.
        If not provided it simply creates a default model with kernel_size of 3 and with 3 layers
        and reads the corresponding file. This might result in an error as you may not already have
        ran a training session for these particular parameters. In this simply run the run_model()
        function and use the returned model for this function.
        
        Parameters
        ----------
            test_loader : DataLoader
                The dataset against which the model needs to be tested.

            model : nn.Module = None
                The model that will be used to plot the confusion matrix against 
                
        Returns
        -------
            np.ndarray
                2D np array representing the confusion matrix.
    '''
    start = time.time()
    # If model not provided create a default model with kernel_size of 3 and 3 layers.
    if not model: 
        model = dta.EmotionCNNKLayers(layers=3).to(device=dta.device)
        
    # test the model by loading the saved model 
    true_labels, output_labels = dta.load_saved_model(model, test_loader=test_loader, path=model_path)
    print("Time taken to test the model: ", (time.time() - start), " seconds.")
    
    # plot the confusion matrix
    confusion_matrix = metrics.confusion_matrix(true_labels, output_labels)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [ "Angry", "Focused", "Neutral", "Happy"])
    cm_display.plot()
    
    plt.savefig(ds.top_level_dir + path + str(model.kernel_size) + "l_"+ str(model.layers)+".png")
    #plt.show()
    return confusion_matrix, "kernel="+str(model.kernel_size) + "_layers=" + str(model.layers)

def run_model(kernel_size: int, layers: int, epochs: int, train_loader: DataLoader, validation_loader: DataLoader, save_path='../models/best_model_test_k') -> nn.Module:
    '''
        This function trains a model with given number of kernel_size and layers. It uses the training
        parameters to run the training method.

        Parameters
        ----------
            None
            
        Returns
        -------
            None
    '''
    print(f"Training model for Kernel Size: {kernel_size} and Layers: {layers}")
    print("Created the CNN model") 
    start = time.time() 
    model = dta.train(train_loader, validation_loader, kernel_size=kernel_size, layers=layers, epochs=epochs, patience_=5, lr=0.001, save_path=save_path)
    print("Time taken to train the model: ", (time.time() - start), " seconds.")
    return model
    
def calculate_evaluation_metrics(confusion_matrix: np.ndarray): 
    '''
        This function calculates the number of true positives, true negatives, false positives, and false negatives
        to find precision, recall, f_measure and accuracy.
    '''
    
    # The true positives are along the diagonal
    TP = np.diag(confusion_matrix)
    print("True Positives: ", TP)
    
    # The false positives are the number images classified in all the other rows except the true row.
    FP = np.sum(confusion_matrix, axis=0) - TP
    print("False Positives: ", FP)
    
    # The false negatives are the number images classifed in all the other columns
    FN = np.sum(confusion_matrix, axis=1) - TP
    print("False Negatives: ", FN)
    
    # The True negatives are the number images in all the other rows and columns except the row and column of the class
    num_classes = 4
    TN  = []
    
    for i in range(num_classes):
        # delete the row of the class from the matrix. 
        temp = np.delete(confusion_matrix, i, 0)
        # delete the column of the class from the matrix.
        temp = np.delete(temp, i, 1)
        TN.append(sum(sum(temp)))
    
    # turn the list into an np.array
    TN = np.array(TN)
    print("True Negatives: ", TN)
    
    # Use the formulas for precision, recall and f_measure to find their values.
    precision = TP/ (TP + FP)
    print("Precision: ", precision)
    
    recall = TP/ (TP + FN)
    print("Recall: ", recall)
    
    f_measure = TP / (TP + (FP + FN)/ 2)
    print("F Measure: ", f_measure)
    
    accuracy = (TP + TN)/ (TP + FN + FP + TN)
    print("Accuracy: ", accuracy)
    
    # Use the Macro and Micro formulas to find Macro and Micro statistics.
    macro_precision = sum(precision) / len(precision)
    print("Macro Precision: ", macro_precision)
    
    macro_recall = sum(recall) / len(recall)
    print("Macro Recall: ", macro_recall)
    
    # find the harmonic mean of the above metrics to find the f_measure.
    macro_f_measure = (2 * macro_precision * macro_recall) / (macro_precision + macro_recall)
    print("Macro F measure: ", macro_f_measure)
    
    micro_precision = sum(TP) / sum(TP + FP)
    print("Micro Precision: ", micro_precision)
    
    micro_recall = sum(TP) / sum(TP + FN)
    print("Micro Recall: ", micro_recall)
    
    # find the harmonic mean of the above metrics to find the f_measure.
    micro_f_measure = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
    print("Micro F measure: ", micro_f_measure)
    
    total_accuracy = sum(TP + TN) / sum(TP + FN + FP + TN)
    
    return macro_precision, macro_recall, macro_f_measure, micro_precision, micro_recall, micro_f_measure, total_accuracy
        


def plot_metrics(metrics: list[list[float]], model_labels: list[str], path="metrics.png"):
    fig, ax = plt.subplots(figsize=(20, 2 + len(model_labels) / 2.5))

    # hide axes
    ax.axis('off')
    ax.axis('tight')

    columns = ['Macro Precision', 'Macro Recall', 'Macro F Measure', 'Micro Precision', 'Micro Recall', 'Micro F Measure', 'Accuracy']
    df = pd.DataFrame(metrics, columns=columns, index=model_labels)
    print(df)
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels = df.index,  loc='center')
    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(16)

    fig.tight_layout()
    plt.plot()
    plt.savefig(path)
    plt.show()

def load_models(models: list[tuple], test_loader: DataLoader):
    model_metrics = []
    model_names = []
    
    for m in models:
        model = dta.EmotionCNNKLayers(kernel_size=m[0], layers=m[1]).to(device=dta.device)
        cm, model_name = plot_confusion_matrix(model=model, test_loader=test_loader)
        t = tuple(map(lambda x: math.ceil(x * 100, 3), calculate_evaluation_metrics(cm)))
        model_metrics.append(t)
        model_names.append(model_name)
    
    plot_metrics(model_metrics, model_names)
    
    
def load_model(kernel_size: int, layers: int, test_loader: DataLoader) -> nn.Module:
    model = dta.EmotionCNNKLayers(kernel_size=kernel_size, layers=layers).to(device=dta.device)
    cm, model_name = plot_confusion_matrix(model=model, test_loader=test_loader)
    t = tuple(map(lambda x: math.ceil(x * 100, 3), calculate_evaluation_metrics(cm)))
    
    plot_metrics([t], [model_name])
    
    return model

def run_model_for_image(model: nn.Module, img_path: str, label: str) -> bool:
    path = ds.top_level_dir + "/part_2/data/"+img_path
    print(path)
    if not os.path.exists(path):
        return False
    
    img = Image.open(path)

    # Convert the image to grayscale
    img = img.convert("L")

    # Convert 48 by 48 image 
    img = img.resize((48, 48), Image.Resampling.LANCZOS)


    # Get the image dimensions
    width, height = img.size

    scoring_dict = {0: "angry", 1: "focused", 2: "neutral", 3: "happy"}
    label_dict = {'angry': 0, "focused" : 1, "neutral": 2, "happy": 3}
    # Extract pixel data
    l = np.asarray(list(img.getdata())).reshape(48, 48).T
    pixels = np.array([l])
    pixels = pixels/255.0
    input_tensor = torch.tensor(pixels, dtype=torch.float32).unsqueeze(1)
    input_labels = torch.tensor(np.array([label_dict[label]]), dtype=torch.long)
    
    input_dataset = TensorDataset(input_tensor, input_labels)
    loader = DataLoader(input_dataset, batch_size = 1, shuffle=False)
    predicted = dta.score_image(model,  loader)
    
    

    
    print("Predicted : "+ scoring_dict[predicted])
    
    return True
  
  
def get_k_fold_dataset_split(k: int, train_size: float = 0.85, validation_size: float = 0.15):
    '''
        This function splits the complete data set csv file k times to 
        return a list of tuples containing the train, validation, and
        test set after each split. 
        
        Parameters
        ----------
            k: int
                The number of times the data must be split.
                
        Returns
        -------
            data : list[tuple[list, list, list]]    
                A list containing tuples where each tuple signifies train, validation, and test data set.
    '''

    data = []
    labels = []
    # Read the complete data set into memory for splitting.
    df = pd.read_csv(ds.complete_path)
    print("Dataset total size: ", len(df))
    
    kf = KFold(k)
    
    i = 0
    for train, test in kf.split(df):
        print("\nFold: ", i + 1, "\n")
        # split the k - 1 folds into training and validation sets
        train_split, validation_split = train_test_split(df.iloc[train], train_size=train_size, test_size=validation_size)
        # get the kth fold using by using the indices in 'test'
        test_split = df.iloc[test]    
        # create the data loader for training
        train_loader, validation_loader, test_loader = dta.create_dataloader(train_split, test_split, validation_split)
        
        model_name = 'fold_num_' + str(i)
        
        # here run_model creates a new model automatically and then trains it.
        model = run_model(3, 3, 10, train_loader, validation_loader, save_path='../models/k_folds/' + model_name)
        
        # We plot and save the confusion matrix
        cm, _ = plot_confusion_matrix(test_loader=test_loader, model=model, path="/part_2/models/k_folds/confusion_matrices/" + model_name, model_path="../models/k_folds/" + model_name)
        # We calculate the evaluation metrics and save them for final table.
        data.append(list(map(lambda x: math.ceil(x * 100, 3), calculate_evaluation_metrics(cm))))
        labels.append(str(i + 1))
        i += 1

    # take the average of each column. 
    avg = [math.ceil(sum_skip_nan(col), 3) for col in zip(*data) ]
    data.append(avg)
    labels.append("Average")
    
    # Finally plot the metrics.
    plot_metrics(data, labels, path=ds.top_level_dir + "/part_2/models/k_folds/metrics.png")
    
def sum_skip_nan(col: list[float]):
    total = 0
    length = 0
    for element in col:
        if not math.isnan(element):
            total += element
            length += 1
            
    return total/length


def calculate_metrics_bias(confusion_matrix: np.ndarray):
    
     # The true positives are along the diagonal
    TP = np.diag(confusion_matrix)
    print("True Positives: ", TP)
    
    # The false positives are the number images classified in all the other rows except the true row.
    FP = np.sum(confusion_matrix, axis=0) - TP
    print("False Positives: ", FP)
    
    # The false negatives are the number images classifed in all the other columns
    FN = np.sum(confusion_matrix, axis=1) - TP
    print("False Negatives: ", FN)
    
    # The True negatives are the number images in all the other rows and columns except the row and column of the class
    num_classes = 4
    TN  = []
    
    for i in range(num_classes):
        # delete the row of the class from the matrix. 
        temp = np.delete(confusion_matrix, i, 0)
        # delete the column of the class from the matrix.
        temp = np.delete(temp, i, 1)
        TN.append(sum(sum(temp)))
    
    # turn the list into an np.array
    TN = np.array(TN)
    print("True Negatives: ", TN)
    
    # Use the formulas for precision, recall and f_measure to find their values.
    precision = TP/ (TP + FP)
    print("Precision: ", precision)
    
    recall = TP/ (TP + FN)
    print("Recall: ", recall)
    
    f_measure = TP / (TP + (FP + FN)/ 2)
    print("F Measure: ", f_measure)
    
    accuracy = (TP + TN)/ (TP + FN + FP + TN)
    print("Accuracy: ", accuracy)
    total_accuracy = sum(TP + TN) / sum(TP + FN + FP + TN)

    for i in range(3):
        if math.isnan(precision[i]):
            precision[i] = 0
        if math.isnan(recall[i]):
            recall[i] = 0
        if math.isnan(f_measure[i]):
            f_measure[i] = 0
        if math.isnan(accuracy[i]):
            accuracy[i] = 0
    
    
    return  sum(precision) / len(precision),  sum(recall) / len(recall), sum(f_measure) / len(f_measure), total_accuracy


def bias_testing (model = "bias"):
    '''
        This function tests the model against the test set and calculates the bias in the model.
        The bias is calculated by finding the difference in the accuracy of the model
    '''


    
    
    # Load the complete dataset    
    
    #
    datasetName = "labeled_data_complete.csv"
    if model == "balanced_bias":
        datasetName = "balanced_data_complete.csv"
    df = pd.read_csv(ds.data_dir_path + datasetName)    

    

    # validate for all y, m, o age groups
    
    # load bias3_l3_n512 model
    df_age_y = df[df['age'] == 'y']
    df_age_m = df[df['age'] ==  'm']
    df_age_o = df[df['age'] == 'o']
    df_gender_m = df[df['gender'] == 'm']
    df_gender_f = df[df['gender'] == 'f']

    
    
    df_age_y_images, df_age_y_labels = dta.preprocess_data(df_age_y)
    df_age_m_images, df_age_m_labels = dta.preprocess_data(df_age_m)
    df_age_o_images, df_age_o_labels = dta.preprocess_data(df_age_o)
    df_gender_m_images, df_gender_m_labels = dta.preprocess_data(df_gender_m)
    df_gender_f_images, df_gender_f_labels = dta.preprocess_data(df_gender_f)
    df_system_images, df_system_labels = dta.preprocess_data(df)
    # Convert numpy arrays to PyTorch tensors
    df_age_y_images = torch.tensor(df_age_y_images, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    df_age_y_labels = torch.tensor(df_age_y_labels, dtype=torch.long)
    df_age_m_images = torch.tensor(df_age_m_images, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    df_age_m_labels = torch.tensor(df_age_m_labels, dtype=torch.long)
    df_age_o_images = torch.tensor(df_age_o_images, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    df_age_o_labels = torch.tensor(df_age_o_labels, dtype=torch.long)
    df_gender_m_images = torch.tensor(df_gender_m_images, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    df_gender_m_labels = torch.tensor(df_gender_m_labels, dtype=torch.long)
    df_gender_f_images = torch.tensor(df_gender_f_images, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    df_gender_f_labels = torch.tensor(df_gender_f_labels, dtype=torch.long)
    df_system_images = torch.tensor(df_system_images, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    df_system_labels = torch.tensor(df_system_labels, dtype=torch.long)

    print("Converted numpy arrays to PyTorch tensors")

    # Create PyTorch datasets
    # TensorDataset  is a Dataset wrapping tensors. Each sample will be retrieved by indexing tensors along the first dimension.
    df_age_y_dataset = TensorDataset(df_age_y_images, df_age_y_labels)
    df_age_m_dataset = TensorDataset(df_age_m_images, df_age_m_labels)
    df_age_o_dataset = TensorDataset(df_age_o_images, df_age_o_labels)
    df_gender_m_dataset = TensorDataset(df_gender_m_images, df_gender_m_labels)
    df_gender_f_dataset = TensorDataset(df_gender_f_images, df_gender_f_labels)
    df_system_dataset = TensorDataset(df_system_images, df_system_labels)


    print("Created PyTorch datasets")
    # Create data loaders 
    # Dataloader is an optimized data loading utility class that helps in efficient data loading and shuffling.
    
    df_age_y_loader = DataLoader(df_age_y_dataset, batch_size=64, shuffle=False)
    df_age_m_loader = DataLoader(df_age_m_dataset, batch_size=64, shuffle=False)
    df_age_o_loader = DataLoader(df_age_o_dataset, batch_size=64, shuffle=False)
    df_gender_m_loader = DataLoader(df_gender_m_dataset, batch_size=64, shuffle=False)
    df_gender_f_loader = DataLoader(df_gender_f_dataset, batch_size=64, shuffle=False)
    df_system_loader = DataLoader(df_system_dataset, batch_size=64, shuffle=False)
    
    #dataloaders for each df

    cm_age_y, _ = plot_confusion_matrix(df_age_y_loader, path="/part_2/models/bias/confusion_matrix_age_y", model_path="../models/"+ model)
    cm_age_m, _ = plot_confusion_matrix(df_age_m_loader,  path="/part_2/models/bias/confusion_matrix_age_m", model_path="../models/"+ model)
    cm_age_o, _ = plot_confusion_matrix(df_age_o_loader,  path="/part_2/models/bias/confusion_matrix_age_o", model_path="../models/"+ model)
    cm_gender_m, _ = plot_confusion_matrix(df_gender_m_loader, path="/part_2/models/bias/confusion_matrix_ gender_m", model_path="../models/"+ model)
    cm_gender_f, _ = plot_confusion_matrix(df_gender_f_loader,  path="/part_2/models/bias/confusion_matrix_ gender_f", model_path="../models/" + model)
    cm_system, _ = plot_confusion_matrix(df_system_loader,  path="/part_2/models/bias/confusion_matrix_system", model_path="../models/"+ model)
   
    metrics_age_y = calculate_metrics_bias(cm_age_y)
    metrics_age_m = calculate_metrics_bias(cm_age_m)
    metrics_age_o = calculate_metrics_bias(cm_age_o)
    metrics_gender_m = calculate_metrics_bias(cm_gender_m)
    metrics_gender_f = calculate_metrics_bias(cm_gender_f)
    metrics_system = calculate_metrics_bias(cm_system)
    data = []
    
    # calculate evaluation metrics for each confusion matrix


    complete_df = pd.read_csv(ds.data_dir_path + datasetName)    
    metrics_age_average = [len(complete_df)]
    for i in range(4):
        metrics_gender_m[i]
        print(f"{metrics_age_y[i]}  {metrics_age_m[i]}  {metrics_age_o[i]}")
        metrics_age_average.append(math.ceil(sum([metrics_age_y[i], metrics_age_m[i],  metrics_age_o[i]])/3* 100))

        

    metrics_gender_average = [len(complete_df)]
    for i in range(4):
        print(metrics_gender_m[i])
        metrics_gender_average.append(math.ceil(sum([metrics_gender_m[i], metrics_gender_f[i]])/2 * 100))



    data.append([len(complete_df[complete_df['age'] == 'y']), math.ceil(metrics_age_y[0] * 100), math.ceil(metrics_age_y[1] * 100), math.ceil(metrics_age_y[2] * 100), math.ceil(metrics_age_y[3] * 100)] )
    data.append([len(complete_df[complete_df['age'] == 'm']), math.ceil(metrics_age_m[0] * 100), math.ceil(metrics_age_m[1] * 100), math.ceil(metrics_age_m[2] * 100), math.ceil(metrics_age_m[3] * 100)] )
    data.append([len(complete_df[complete_df['age'] == 'o']), math.ceil(metrics_age_o[0] * 100), math.ceil(metrics_age_o[1] * 100), math.ceil(metrics_age_o[2] * 100), math.ceil(metrics_age_o[3] * 100)] )
    
    
    data.append(metrics_age_average)
    data.append([len(complete_df[complete_df['gender'] == 'm']), math.ceil(metrics_gender_m[0] * 100), math.ceil(metrics_gender_m[1] * 100), math.ceil(metrics_gender_m[2] * 100), math.ceil(metrics_gender_m[3] * 100)] )
    data.append([len(complete_df[complete_df['gender'] == 'f']), math.ceil(metrics_gender_f[0] * 100), math.ceil(metrics_gender_f[1] * 100), math.ceil(metrics_gender_f[2] * 100), math.ceil(metrics_gender_f[3] * 100)] )
    
    data.append(metrics_gender_average)
    data.append([len(complete_df), math.ceil(metrics_system[0] * 100), math.ceil(metrics_system[1] * 100), math.ceil(metrics_system[2] * 100), math.ceil(metrics_system[3] * 100)] )

    
    # Construct a table with the metrics
    fig, ax = plt.subplots(figsize=(20, 2 + 5 / 2.5))

    # for i in range(5):
    #     for j in range(4):
    #         data[i][j] = str(data[i][j]) + "%"

    # hide axes
    ax.axis('off')
    ax.axis('tight')

    columns = ['#Image','Precision', 'Recall', 'F1 Score', 'Accuracy']

    

    table = ax.table(cellText=data, colLabels=columns, rowLabels = ['Young', 'Middle-Aged', 'Senior', 'Average Age', 'Male', "Female", 'Average Gender', 'System Metrics'],  loc='center')
    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(16)

    fig.tight_layout()
    plt.plot()
    plt.savefig(ds.top_level_dir + "/part_2/models/bias/metrics.png")
    plt.show()


    # print("Metrics for age group y: ", metrics_age_y)
    # print("Metrics for age group m: ", metrics_age_m)
    # print("Metrics for age group o: ", metrics_age_o)
    


    
if __name__ == "__main__":
    
    choice = input("Do you want to perform bias testing for evaluation? (y/n): ")
    if choice == 'y':
       

        choice = input("For seeing the metrics press 1, to generate table on biased models press 2: ")
        if choice == '1':
            
            choice = input("To retrain the models, write 1, to load the models, write 1: ")
            if choice == '1':
                ds.split_dataset( train_size=0.7, test_size=0.15)
                train_data, validation_data, test_data = ds.get_dataset(train_path=ds.data_dir_path + "labeled_data_train.csv", test_path=ds.data_dir_path + "labeled_data_validation.csv", validation_path=ds.data_dir_path + "labeled_data_test.csv")
                train_loader, validation_loader, test_loader = dta.create_dataloader(train_data, test_data, validation_data)
                run_model(3, 3, 10, train_loader, validation_loader, save_path='../models/bias')

            bias_testing()
        if choice == '2':

            
            choice = input("To retrain the model, write 1; To use the existing model, write 2:")
            
            if choice == '1':
                #initial 12508
                men = 12508
                #initial : 7492
                women = 6000

                #generate new dataset that respects this number 

                # Load the complete dataset
                df = pd.read_csv(ds.data_dir_path + "labeled_data_complete.csv")

                male_df = df[df['gender'] == 'm']
                female_df = df[df['gender'] == 'f']

                sampled_male_df = male_df.sample(n=men, random_state=1)
                sampled_female_df = female_df.sample(n=women, random_state=1)

                balanced_df = pd.concat([sampled_male_df, sampled_female_df])

                balanced_df.to_csv("../data/balanced_data_complete.csv", index=False)

                ds.split_dataset( train_size=0.7, test_size=0.15)
                train_data, validation_data, test_data = ds.get_dataset(train_path=ds.data_dir_path + "balanced_data_train.csv", test_path=ds.data_dir_path + "balanced_data_validation.csv", validation_path=ds.data_dir_path + "balanced_data_test.csv")
                train_loader, validation_loader, test_loader = dta.create_dataloader(train_data, test_data, validation_data)
                run_model(3, 3, 10, train_loader, validation_loader, save_path='../models/balanced_bias')
            bias_testing("balanced_bias")


            
    else:       
        choice = input("Do you want to perform k fold cross validation? (y/n): ")
        if choice == 'y':
            k = int(input("Please input the value for k: "))
            get_k_fold_dataset_split(k)

            
        else:
        
            train_loader, validation_loader, test_loader = dta.create_dataloader(ds.get_dataset())

            while True:
                choice = input("Please input a command: \n1. Train Model\n2. Load Model\n3. Split Data\n4. Exit\nYour choice: ")
                if choice == "1":
                    print()
                    kernel_size = int(input("Please input the kernel size: "))
                    layers = int(input("Please input number of layers: "))
                    epochs = int(input("Please input the number epochs: "))

                    model = run_model(kernel_size, layers, epochs, train_loader, validation_loader)

                    print()
                    while True:

                        choice = input("Do you want to run the model for an input image(y/n)? ")
                        if choice == "y":
                            class_ = input("Please input the class of image(angry, focused, neutral, happy): ")

                            if class_ == "angry" or class_ == "happy" or class_ == "neutral" or class_ == " focused":
                                img = input("Name of image(as .jpg): ")
                                if not run_model_for_image(model, class_ + "/" + img, class_):
                                    print("Could not find the image!")


                        else:
                            print("Going back to menu")
                            break
                        
                elif choice == "2":
                    print()
                    kernel_size = int(input("Please input the kernel size: "))
                    layers = int(input("Please input number of layers: "))

                    model = load_model(kernel_size, layers, test_loader)

                    print()

                    while True:

                        choice = input("Do you want to run the model for an input image(y/n)? ")
                        if choice == "y":
                            class_ = input("Please input the class of image(angry, focused, neutral, happy): ")

                            if class_ == "angry" or class_ == "happy" or class_ == "neutral" or class_ == " focused":
                                img = input("Name of image(as .jpg): ")
                                if not run_model_for_image(model, class_ + "/" + img, class_):
                                    print("Could not find the image!")      

                        else:
                            print("Going back to menu")
                            break
                elif choice == "3":
                    print()
                    train_size = input("Please input the train size: ")
                    test_size = input("Please input the test_size: ")

                    ds.split_dataset(float(train_size), float(test_size))
                    print()
                elif choice == "4":
                    print("Goodbye.")
                    break
                else:
                    print("Wrong choice.")