#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 08:13:05 2020

@author: Benjamin
"""

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

"""
Function that splits a large array into windows of given size and overlap.
"""
def windowed_view(arr, window, overlap):
    '''https://stackoverflow.com/questions/18247009/window-overlap-in-pandas/18247903#18247903'''
    arr = np.asarray(arr)
    if arr.ndim > 1:
        arr = arr.flatten()
    padding = [np.NaN] * (window - 1)
    arr = np.asarray(list(itertools.chain(padding, arr, padding)))
    window_step = window - overlap
    new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step, window)
    new_strides = (arr.strides[:-1] + (window_step * arr.strides[-1],) + arr.strides[-1:])
    return np.lib.stride_tricks.as_strided(arr, shape=new_shape, strides=new_strides, writeable=True)

directory = "/Users/Benjamin/OneDrive - University of Strathclyde/EE475 Project/Data/PECAN Street/"
folder = "1minute_all_ev_data/"
file = "1minute_ev_data_"
# List of all houses w/ non-null EV power data.
house_ids = ['27', '661', '1222', '1642', '3000', '4373', '4767', '5679', '6139', '7719', '8156', '9053']

# List to store Dataframes for each houses data.
all_windows = []

# Read in each houses data into Dataframes from csv files.
for x in house_ids:
    
    house_df = pd.read_csv(directory + folder + file + x + '.csv')
    
    # Removing redundant columns & redundant timestamp accuracy.
    house_df = house_df.drop(columns=['Unnamed: 0'])
    house_df.localminute = house_df.localminute.apply(lambda x : x[:-6])
    # Making sure data is ordered correctly.
    house_df = house_df.sort_values(by='localminute')
    
    # Splitting date/time, agreggate & submetered EV power data into 10 minute overlapping windows.
    grid_windows = windowed_view(house_df.grid, 10, 9)
    ev_windows = windowed_view(house_df.car1, 10, 9)
    date_windows = windowed_view(house_df.localminute, 10, 9)
    
    # Lists to store charging classifications & timestamps of windows.
    charging = []
    timestamps = []
    
    index = 0
    for w in ev_windows:
        
        # Classify window as 'EV charging' if submetered power > 0.25kW
        for i in range(0, 10):
            if w[i] > 0.25:
                charging.append(1)
                break
            elif i == 9:
                charging.append(0)
        
        timestamps.append(date_windows[index][0])
        index += 1
    
    # Convert windows, classifications and timestamps to Dataframe.
    window_df = pd.DataFrame().from_records(grid_windows)
    window_df['Classification'] = charging
    window_df['Timestamp'] = timestamps
    window_df['Data ID'] = int(x)
    # Select equal number of Charging & Disconnected windows from current house.
    window_df = window_df.sort_values(by='Classification', ascending=False)
    class_counts = window_df.Classification.value_counts()
    window_df = window_df.head(class_counts[1] * 2)
    # Add Dataframe to list.
    all_windows.append(window_df)
    
# Combine all data into one Dataframe.
window_df = pd.concat(all_windows, ignore_index = True)
window_df.reset_index(drop=True, inplace=True)
# Drop null rows.
window_df = window_df.dropna()
print(len(window_df))

pca_start = datetime.datetime.now()

# Standardise data and apply PCA.
standard = StandardScaler().fit_transform(window_df[window_df.columns[0:10]])
pca = PCA(n_components=4)
principal_components = pca.fit_transform(standard)

# Timing PCA & finding percentage of explained variance of components.
pca_time = datetime.datetime.now() - pca_start
print(pca_time)
var_percent = pca.explained_variance_ratio_
# print(var_percent)


# Split data into train & test sets (80:20)
# Using windowed active power directly for model.
labels = window_df.Classification
train, test, train_labels, test_labels = train_test_split(window_df[window_df.columns[0:10]], labels, test_size=0.2)

# Alternatively, uncomment following to use PCs as input to model.
# train, test, train_labels, test_labels = train_test_split(principal_components, labels, test_size=0.2)

print(train.shape, test.shape)
print(train_labels.value_counts())
print(test_labels.value_counts())

# Create Random Forest model with 500 trees.
rf_model = RandomForestClassifier(n_estimators = 500,
                                max_features = None,
                                criterion = 'entropy',
                                random_state = 50,
                                n_jobs = -1, verbose = 0)

rf_start = datetime.datetime.now()

# Fit on training data
rf_model.fit(train, train_labels)
train_time = datetime.datetime.now() - rf_start

# Test
rf_predictions = rf_model.predict(test)
test_time = datetime.datetime.now() - rf_start - train_time
rf_probs = rf_model.predict_proba(test)[:, 1]

print(train_time)
print(test_time)

# Generate confusion matrix from results and calculate F1 score.
cm = confusion_matrix(test_labels, rf_predictions)
f_score = 2*cm[1][1] / (2*cm[1][1] + cm[1][0] + cm[0][1])
print(f_score)


# Plotting 2D Representation of Data
# =============================================================================
#         
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 Dimensional PCA', fontsize = 20)
# 
# targets = [0, 1]
# colors = ['r', 'g']
# for target, color in zip(targets,colors):
#     indicesToKeep = principal_components['Classification'] == target
#     ax.scatter(principal_components.loc[indicesToKeep, 0]
#                , principal_components.loc[indicesToKeep, 1]
#                , c = color)
#                , s = 50)
# ax.legend(targets)
# ax.grid()
# =============================================================================

# Plotting Explained Variance of Each Component
# =============================================================================
# xaxis = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']
# 
# for i in range(len(var_percent)):
#     var_percent[i] = var_percent[i]*100
# 
# fig, ax1 = plt.subplots()
# 
# PC_bar = ax1.bar(xaxis, var_percent, 0.5, color='tab:blue')
# 
# for index, value in enumerate(var_percent):
#     plt.text(index-0.36, value+0.95, str(round(value, 2)), size=9)
# 
# ax1.set_ylabel('Percentage of Explained Variance')
# ax1.set_xlabel('Principal Component')
# 
# plt.grid(color='gainsboro', linestyle='dashed')
# fig.set_size_inches(5, 3)
# plt.show()
# =============================================================================
