import mne
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math
import argparse


from features.build_features import get_eeg_data, get_padded_x_and_y, get_segments, get_X_and_y

from models.models import CNN, MaskedLSTM, TransformerModel



def arg_parser():
    parser = argparse.ArgumentParser(description='EEG-Gait')
    parser.add_argument('--set_file', type=str, default='data/raw/EEG_data.csv',
                        help='path to the csv file containing the eeg data')
    parser.add_argument('--time_file', type=str, default='data/raw/EEG_time.csv',
                        help='path to the csv file containing the time data')

    parser.add_argument('--stride', type=list, default=[0.1,0.1],
                        help='stride of the lag window')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--model", type=str, default="lstm")
    parser.add_argument("--single_patient", type=bool, default=False)
    parser.add_argument("--participant_path", type=str, default="Z:/Projects/NCM/NCM_StimuLOOP/project_only/03_NeuroFeedback/project_only/02_Data/EEG/filtered")
    

    return argparse.parse_args()


def plots(x_axis,t):
    """
    Plots the accuracy or loss over the epochs
    Parameters
    ----------
    x_axis : list
        List of values to plot.
    t : str
        Title of the plot.
    Returns
    -------
    None.
    """
    plt.plot(x_axis)
    plt.title(t)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy" if "accuracy" in t else "Loss")
    plt.show()


def multiparticipant_x_and_y(args):
    participant_path = args.participant_path
    participant_list = [directory for directory in os.listdir(participant_path)]

    #remove first two participants
    participant_list.pop(0)
    participant_list.pop(0)
    #select trial type you want to have

    #trial_type = ["posttreadmill_new","pretreadmill_new"]
    trial_type = ["posttreadmill"]

    all_segments_left = list()
    all_labels_left = list()
    all_segments_right = list()
    all_labels_right = list()

    for trial in trial_type:
        for index,participant in enumerate(participant_list):
            data_path = participant_path +"/"+ participant + "/ProcessedData/EEG/filtered/" + trial + "_new.set"
            raw = mne.io.read_raw_eeglab(data_path, preload=True)
            data = raw.to_data_frame()
            data = pd.concat([data,data.rolling(window=2).mean()],axis =0)
            data = data.sort_values("time").reset_index(drop=True)
            data = data.drop(data.iloc[-1].name)
            #Load corresponding Timefile of Participant
            time_file = "Z:/Projects/NCM/NCM_StimuLOOP/project_only/03_NeuroFeedback/project_only/04_Results/times_in_seconds/posttreadmill/"+participant+"_"+trial+"_timeframes_in_seconds.csv"
            df_time = pd.read_csv(time_file)
            
            print("getting the segment")
            segments,left,right,mixed = get_segments(data,df_time)
            if left:
                labels = [0 if i % 2 == 0 else 1 for i in range(len(segments))]
                all_segments_left.extend(segments)
                all_labels_left.extend(labels)
                
            if right:
                labels = [2 if i % 2 == 0 else 3 for i in range(len(segments))]
                all_segments_right.extend(segments)
                all_labels_right.extend(labels)
                
    print("finding all left labels, and Segments")
    X_left,y_left = get_X_and_y(all_segments_left,all_labels_left)
    print("finding all right labels and Segments")
    X_right,y_right = get_X_and_y(all_segments_right,all_labels_right)

    len_left = X_left.shape[1]
    len_right = X_right.shape[1]

    max_len = max(len_left, len_right)

    if len_left < max_len:
        padding_length = max_len - len_left
        X_left = np.pad(X_left, ((0, 0), (0, padding_length), (0, 0)), mode='constant', constant_values=0)

    elif len_right < max_len:
        padding_length = max_len - len_right
        X_right = np.pad(X_right, ((0, 0), (0, padding_length), (0, 0)), mode='constant', constant_values=0)

    X = np.concatenate((X_left, X_right), axis=0)
    y = np.concatenate((y_left ,y_right), axis=0)
    return X,y

if __name__ == "__main__":


    args = arg_parser()


    if args.single_patient:
    #get the whole dataset of eeg data as a df
        final, timewindow = get_eeg_data(args.set_file, args.time_file, timeshift=[0.1,0.1] if args.stride is None else args.stride)

        #get the padded data X and class label y
        X, y = get_padded_x_and_y(final, timewindow, args.window_size, args.stride)
    else:
        #get list of all particpants
        X, y = multiparticipant_x_and_y(args)


    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2,shuffle = False)

    # Create DataLoader objects for training and validation sets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32,shuffle = False)


    #training is possible with a gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #€ Set device, input size, hidden size, number of layers, and number of classes for LSTM
    input_size = X.shape[2]
    hidden_size = 64
    num_layers = 1
    num_classes = len(np.unique(y))

    # Create the model and move it to the device
    if args.model == "cnn":
        model = CNN(input_size, num_classes).to(device)
    elif args.model == "lstm":
        model = MaskedLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    elif args.model == "transformer":
        model = TransformerModel().to(device)


    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)



    # Training the model
    num_epochs = 100

    overtime=list()
    tot_loss = list()

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluate the model on the validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(device)
                labels = labels.to(device)
                
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Accuracy: {val_acc:.4f}")
        
        overtime.append(val_acc)
        tot_loss.append(loss.item())

    plots(overtime,"Validation Accuracy")
    plots(tot_loss,"Loss")
