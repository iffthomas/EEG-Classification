# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

import os 
import re
import argparse


def get_cummulative_frames(df,name_start,trial_path,trial_name):
    """Extract the start and end frames of the data frame we read in
    df: Is a Dataframe we want to process with the columns name_start, name_end
    name_start: Is a string that indicates the starting column of the data
    name_end: Is a string that indicates the end column of the data
    The Data used was an excel file that extracted for Individual GaitParameters/GaitEvents of cutted and uncutted walking trials
    Code here can be found under the name Filtering Post in
    """
    #---------------------------------------------------------------------------------------------------------------------------"
    ### two column vectors that ignore all the nan entries, one with the mat structure, one from the cutted manually timepoints"
    
    name_start_cut = df[name_start][~np.isnan(df[name_start])].reset_index(drop=True)
    
    trials = pd.read_csv(trial_path,sep=";")
    
    current_frames = trials[trial_name].to_numpy()
    current_frames=current_frames[~np.isnan(current_frames)]
    
    assert len(df["startcut"][~np.isnan(df["startcut"])]) == len(current_frames), "startcut & trial dont have same dimensions, it appears that individual trials must have been deleted"
    
    name_start_cut = (df["startcut"][~np.isnan(df["startcut"])] +current_frames).to_numpy() #combine the two dataframes that we get the exact start time. 
    

    #Use Backtransform function to get all the elements in the right index of our actual dataframe" 
    d,i=0,0
    while i<len(df[name_start]):
        if ~np.isnan(df[name_start][i]):
            df[name_start][i]=name_start_cut[d]
            d+=1
        i+=1
    return df


def add_start_time_to_column(df,colname):
    """Transforms a column by adding the right starttimeframe for each desired column
    df["start" ] is here that we don't overwrite startcut for possible other transformations
    df: our needed Data.Frame
    colname: name of a column we want to add the start time"""
    #print("{} is getting the right timeframes".format(colname))


    df["start"]=df["startcut"]
    indexes = df["start"][~np.isnan(df["start"])].index

    start_list = list()
    for index in range(1,len(indexes)):
        if index ==len(indexes)-1: #to get the last two entries to the start_list
            row = df["start"][indexes[index-1]:indexes[index]].reset_index(drop=True) #add second element and interpolate
            for _ in row:
                start_list.append(row[0])
            row = df["start"][indexes[-1]:].reset_index(drop=True) #add last element and interpolate
            for _ in row:
                start_list.append(row[0])
        else:
            row = df["start"][indexes[index-1]:indexes[index]].reset_index(drop=True)
            for _ in row:
                start_list.append(row[0])
    start_list = np.asarray(start_list)
    assert df["start"].size == len(start_list), "The start_list produces a wrong Output, please double Check in preprocessing.py the function add_start_time_to_column"
    del df["start"]
    df["start_list"]= start_list
    df[colname]+=df["start_list"] #add the start list to the required column.
    


def check_nans(df, liste):
    """Checks if there are any nans in the columns of the dataframe
    df: Dataframe we want to check
    liste: list of columns we want to check for nans
    """

    for column in liste:
        df = df[~np.isnan(df[column])]
    return df

def filt(df,liste):
    """Filters the dataframe for outliers
    df: Dataframe we want to filter
    liste: list of columns we want to check for outliers
    """
    from scipy import stats
    for column in liste:
        if len(df[column]) > 5:
            mad = stats.median_abs_deviation(df[column],axis = None)
            median = np.nanmedian(df[column])
            df=df[(df[column] > median - 4*mad) | (df[column] < median + 4*mad) ]
    return df

def arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--participant', type=str, default="P1",
                        help='Participant name')
    parser.add_argument('--trial', type=str, default="obstacle",
                        help='Trial name')
    parser.add_argument('--path', type=str, default="Z:/Projects/NCM/NCM_StimuLOOP/project_only/03_NeuroFeedback/project_only/02_Data/Young_NFB/",
                        help='Path to the data')
    parser.add_argument('--path_cutting_trials', type=str, default="Z:/Projects/NCM/NCM_StimuLOOP/project_only/03_NeuroFeedback/project_only/04_Results/times_to_cut/",
                        help='Path to the cutting trials')
    parser.add_argument('--path_cutting_trials_seconds', type=str, default="Z:/Projects/NCM/NCM_StimuLOOP/project_only/03_NeuroFeedback/project_only/04_Results/times_in_seconds/",
                        help='Path to the cutting trials in seconds')
    return parser.parse_args()


    
    
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    args = arg_parser()

    path_participants = args.path
    participant = args.participant

    participant_list = [directory for directory in os.listdir(path_participants)]
    participant_list.pop(0)
    to_transform = ["FFleftlocs","FFrightlocs","HSleftlocs","HSrightlocs",
    "RmidStlocs","RmidSwlocs","MTCrightlocs","MTCleftlocs","LmidStlocs","LmidSwlocs","TOleftlocs","TOrightlocs"]
    targets= ["FFleftlocs","FFrightlocs","HSleftlocs","HSrightlocs",
    "RmidStlocs","RmidSwlocs","LmidStlocs","LmidSwlocs","TOleftlocs","TOrightlocs","start_list","finish_list"]
    for participant in participant_list:
        folderpath = path_participants+participant+"/ProcessedData/Vicon/timeseries/"
        files = os.listdir(folderpath)
        filenames =[name for name in files if name.endswith(".csv")]
        trial_names = [trial for trial in filenames]
        
        #get all trialnames, obstacle, pretreadmill,etc.
        for i,trial in enumerate(trial_names):
            position = re.search("_",trial).start()
            trial_names[i]=trial_names[i][:position]
        print("Particpant: {}".format(participant))
        #iteratre over df_path
        for j,filename in enumerate(filenames):
            
            df_path = folderpath+filename
            trial_path = path_participants+participant+"/ProcessedData/Vicon/cutted_trials_start.csv"
            current_trial = trial_names[j]
            print("Trial: {}".format(current_trial))
            
            df = pd.read_csv(df_path)
            if current_trial in ["obstacle","pre8walk","post8walk"]:
                df = get_cummulative_frames(df,"startcut",trial_path,current_trial)
                
                for name in to_transform:
                    add_start_time_to_column(df,name)
                
                df["finish_list"]=df["start_list"]+df["finshcut"]
            else:
                df["start_list"] = df["startcut"]
                df["finish_list"] = df["finshcut"]
            nan_list = [
    "LmidStlocs",
    "LmidSwlocs",
    "RmidStlocs",
    "RmidSwlocs",
    "dlsL",
    "dlsR",
    "durationGaitCycleL",
    "durationGaitCycleR",
    "durationStancePhL",
    "durationStancePhR",
    "durationSwingPhL",
    "durationSwingPhR",
    "midStOccTimeL",
    "midStOccTimeR",
    "midSwOccTimeL",
    "midSwOccTimeR",
    "ndlsL",
    "ndlsR",
    "stepLengthL",
    "stepLengthR",
    "stepTimeL",
    "stepTimeR",
    "stepWidthL",
    "stepWidthR",
    "strideLengthL",
    "strideLengthR",
    "FFleftlocs","FFrightlocs","HSleftlocs","HSrightlocs", "HSleft","HSright","TOleft","TOleftlocs","TOright","TOrightlocs"]
            
            liste = [
    "HSleft","HSright","TOleft","TOright","ndlsL",
    "ndlsR",
    "stepLengthL",
    "stepLengthR",
    "stepTimeL",
    "stepTimeR",
    "stepWidthL",
    "stepWidthR",
    "strideLengthL",
    "strideLengthR",
    "durationGaitCycleL",
    "durationGaitCycleR",
    "durationStancePhL",
    "durationStancePhR",
    "durationSwingPhL",
    "durationSwingPhR","ndlsL",
    "ndlsR","dlsL",
    "dlsR","midStOccTimeL",
    "midStOccTimeR",
    "midSwOccTimeL",
    "midSwOccTimeR",
            ]
            
            df = check_nans(df,nan_list)
            #if current_trial != "obstacle":
                #df= filt(df,liste)
            df=df.reset_index(drop=True)
            
            
            
            #select all timepoints we need to cut the data
            target_df = df[targets]
            target_path = "Z:/Projects/NCM/NCM_StimuLOOP/project_only/03_NeuroFeedback/project_only/04_Results/times_to_cut/{}/{}_{}_timeframes.csv".format(current_trial,participant,current_trial,sep = ";")
            target_df.to_csv(target_path, index = False) #save this data
            
            #save the whole combination of gaitevents, gaitparameters. 
            path = "Z:/Projects/NCM/NCM_StimuLOOP/project_only/03_NeuroFeedback/project_only/04_Results/Individual_GaitParameters/{}/{}_{}_summary.csv".format(current_trial,participant,current_trial,sep = ";")
            df.to_csv(path,index=False)
            

    path_cutting_trials = args.path_cutting_trials
    trial_type = [directory for directory in os.listdir(path_cutting_trials)]



    #dirty fix since data was not recorded in a good way
    frequencies = [[100 for x in range(0,15)],
    [200,100,200,200,100,100,100,200,100,100,100,100,100,200,100],
    [200 for x in range(0,15)],
    [100 for x in range(0,15)],
    [200 for x in range(0,15)]]


    for j,trial in enumerate(trial_type):
        freq_relevant = frequencies[j]
        filenames = [name for name in os.listdir(path_cutting_trials + trial) if name.endswith(".csv")]
        for i,file in enumerate(filenames):
            freq = freq_relevant[i]
            trial_path = path_cutting_trials+"/"+trial+"/"+file
            df_timeframes = pd.read_csv(trial_path,sep = ",")
            df_timeframes = df_timeframes / freq
            name = filenames[i][0:re.search(".csv",filenames[0]).start()]
            save_path = args.path_cutting_trials_seconds + trial +"/" + name+"_in_seconds.csv"
            df_timeframes.to_csv(save_path,index = False)
    

    
    
        

