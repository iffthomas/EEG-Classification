
import pandas as pd
import numpy as np
import mne
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fftshift
from scipy import stats


def get_eeg_data(set_file, time_file, timeshift=[0.1,0.1]):
    """Extracts the eeg data from the set file and the time file and returns a dataframe with the eeg data and the time in seconds.
    Parameters
    ----------
    set_file : str
        Path to the set file.
    time_file : str
        Path to the time file.
    Returns
    -------
    final : pd.DataFrame
        Dataframe with the eeg data and the time in seconds.
    """

    raw = mne.io.read_raw_eeglab(set_file, preload=True)
    df = raw.to_data_frame()


    time_file = "Z:/Projects/NCM/NCM_StimuLOOP/project_only/03_NeuroFeedback/project_only/04_Results/times_in_seconds/posttreadmill/P4-0004_posttreadmill_timeframes_in_seconds.csv"
    df_time = pd.read_csv(time_file)
    df = pd.concat([df,df.rolling(window=2).mean()],axis =0)
    df = df.sort_values("time").reset_index(drop=True)
    df = df.drop(df.iloc[-1].name)

    # to find if it start from left or right & check if there is step missing
    x = df_time["HSrightlocs"] - df_time["HSleftlocs"] > 0
    a =x.value_counts()
    if len(a.index) == 1 and a.values[0] == len(x) and x.all():
        print("this trial always starts with a left HeelStrike")
        timewindow = df_time[["HSleftlocs","TOleftlocs","TOrightlocs"]]
        timewindow["time"] = df_time["HSleftlocs"]
    elif len(a.index)==1 and a.values[0] == len(x): 
        print("this trial always starts with a right HeelStrike")
        timewindow = df_time[["HSrightlocs","TOrightlocs","TOleftlocs"]]
        timewindow["time"] = df_time["HSrightlocs"]
    else:
        print("we have alternating starting gaitcycles and steps are therefore missing, we have to refine our strategy")

    ## to adjust the time window if needed
    timewindow=timewindow.drop(columns="time")
    extracted_data = pd.DataFrame()
    timewindow["end_Hsleft"] = timewindow["TOleftlocs"] + timeshift[0]


    for index, row in timewindow.iterrows():
        start_event0 = row[0]
        end_event0= row[3]
        start_event1 = row[1]
        end_event1 = row[2]

        event0_data = df[(df['time'] >= start_event0) & (df['time'] < end_event0)].copy()
        event1_data = df[(df['time'] >= start_event1) & (df['time'] < end_event1)].copy()

        extracted_data = pd.concat([extracted_data,event0_data])
        extracted_data = pd.concat([extracted_data,event1_data])


    extracted_data.reset_index(drop=True, inplace=True) 
    final = extracted_data.copy()
    final = final.drop(columns=["time"])

    final["time"] = extracted_data["time"]



    #get the timeshift for the eeg data
    timewindow["end_Hsleft"] = timewindow["TOleftlocs"] +timeshift[0]
    timewindow["TOrightlocs"] += timeshift[1]

    return final, timewindow

def get_padded_x_and_y(final, timewindow):
    time_points=list()
    for index, row in timewindow.iterrows():
        tup_odd = (row[0],row[3])
        tup_even = (row[1],row[2])
        time_points.append(tup_odd)
        time_points.append(tup_even)
    

# Extract the time series segments
    segments = [final[(final["time"] >= start) & (final["time"] <= end)] for start, end in time_points]

# Pad the segments to have the same length
    max_length = max([segment.shape[0] for segment in segments])
    padded_segments = []
    padded_labels = []

    for idx, segment in enumerate(segments):
        padding_length = max_length - segment.shape[0]
        padded_data = np.pad(segment.iloc[:, 1:-1].values, ((0, padding_length), (0, 0)), mode='constant', constant_values=0)
        padded_segments.append(padded_data)
    
    # Assigning labels based on the alternating pattern
        label = 0 if idx % 2 == 0 else 1
        padded_labels.append(label)

# Stack the padded segments(x) and labels(y)
    X = np.stack(padded_segments)
    y = np.array(padded_labels)

    return X, y


def get_segments(df,df_time):
    x = df_time["HSrightlocs"] - df_time["HSleftlocs"] > 0
    a =x.value_counts()
    timewindow =pd.DataFrame()
    left,right,mixed = False,False,False
    if len(a.index) == 1 and a.values[0] == len(x) and x.all():
        print("this trial always starts with a left HeelStrike")
        timewindow = df_time[["HSleftlocs","TOleftlocs","TOrightlocs"]]
        timewindow["time"] = df_time["HSleftlocs"]
        timewindow["end_hsleft"] = timewindow["TOleftlocs"] + 0.1
        timewindow["TOrightlocs"]+= 0.1
        left = True
    elif len(a.index)==1 and a.values[0] == len(x): 
        print("this trial always starts with a right HeelStrike")
        timewindow = df_time[["HSrightlocs","TOrightlocs","TOleftlocs"]]
        timewindow["time"] = df_time["HSrightlocs"]
        timewindow["end_hsright"] = timewindow["TOrightlocs"] + 0.1
        timewindow["TOleftlocs"] += 0.1
        right = True
    else:
        print("we have alternating starting gaitcycles and steps are therefore missing, we have to refine our strategy")
        mixed = True

    timewindow=timewindow.drop(columns="time")
    extracted_data = pd.DataFrame()
    for index, row in timewindow.iterrows():
        start_event0 = row[0]
        end_event0= row[3]
        start_event1 = row[1]
        end_event1 = row[2]

        event0_data = df[(df['time'] >= start_event0) & (df['time'] < end_event0)].copy()
        event1_data = df[(df['time'] >= start_event1) & (df['time'] < end_event1)].copy()

        extracted_data = pd.concat([extracted_data,event0_data])
        extracted_data = pd.concat([extracted_data,event1_data])


    extracted_data.reset_index(drop=True, inplace=True) 
    final = extracted_data.copy()
    final = final.drop(columns=["time"])
    final["time"] = extracted_data["time"]
    time_points=list()
    for index, row in timewindow.iterrows():
        tup_odd = (row[0],row[3])
        tup_even = (row[1],row[2])
        time_points.append(tup_odd)
        time_points.append(tup_even)
    # Extract the time series segments
    segments = [final[(final["time"] >= start) & (final["time"] <= end)] for start, end in time_points]


    return segments,left,right,mixed

        
def get_X_and_y(all_segments,all_labels):
    max_length = max([segment.shape[0] for segment in all_segments])
    padded_segments = []
    padded_labels = []

    for idx, segment in enumerate(all_segments):
        padding_length = max_length - segment.shape[0]
        padded_data = np.pad(segment.iloc[:, :-1].values, ((0, padding_length), (0, 0)), mode='constant', constant_values=0)
        padded_segments.append(padded_data)

        # Copy the corresponding label
        padded_labels.append(all_labels[idx])

    # Stack the padded segments and labels
    X = np.stack(padded_segments)
    y = np.array(padded_labels)

    return X,y

