#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 09:43:53 2021

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
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from functions import *

# List of all houses w/ non-null EV power data.
house_ids = ['27', '661', '1222', '1642', '3000', '4373', '4767', '5679', '6139', '8156', '9053']

window_size = 6
overlap = 5
# List to store Dataframes for each houses data.
all_train = []
all_test = []

# Read in each houses data into Dataframes from csv files.
for x in house_ids:
    
    window_df = csv_data_to_windows(x, window_size, overlap, 15)
    
    # Select equal number of Charging & Disconnected windows from current house.
    # Drop null rows.
    window_df = window_df.dropna()
    print(len(window_df))
    
    # Balance by oversampling (SMOTE).
    window_df = window_df.drop('Timestamp', axis=1)
    disconnected = window_df[window_df.Classification == 0]
    disconnected_len = disconnected.shape[0]
    charging = window_df[window_df.Classification == 1]
    charging_len = charging.shape[0]
    ratio = charging_len/disconnected_len
    
    disconnected_test = disconnected.sample(frac=ratio*0.2)
    disconnected = disconnected.drop(disconnected_test.index)
    charging_test = charging.sample(frac=0.2)
    charging = charging.drop(charging_test.index)
    test_df = pd.concat([charging_test, disconnected_test], ignore_index = True)
    test_df = test_df.sample(frac=1)
    test_df.reset_index(drop=True, inplace=True)
    
    class_counts = test_df.Classification.value_counts()
    print(class_counts)

    train_df = pd.concat([charging, disconnected], ignore_index = True)
    train_df = train_df.sample(frac=1)
    train_df.reset_index(drop=True, inplace=True)
    
    train_df = oversample(train_df, window_size)
    
    class_counts = train_df.Classification.value_counts()
    print(class_counts)
    
    # Add Dataframe to list.
    train_df['Data ID'] = int(x)
    test_df['Data ID'] = int(x)
    all_train.append(train_df)
    all_test.append(test_df)
    
# Combine all data into one Dataframe.
train_df = pd.concat(all_train, ignore_index = True)
train_df.reset_index(drop=True, inplace=True)
test_df = pd.concat(all_test, ignore_index = True)
test_df.reset_index(drop=True, inplace=True)
# Drop null rows.
window_df = window_df.dropna()
print(len(window_df))

class_counts = train_df.Classification.value_counts()
print(class_counts)

pc_df = apply_pca(train_df, window_size)

# Split data into train & test sets (80:20) using windowed active power directly for input to  model.
# =============================================================================

train = train_df[train_df.columns[0:window_size]]
test = test_df[test_df.columns[0:window_size]]
train_labels = train_df.Classification
test_labels = test_df.Classification
# train, test, train_labels, test_labels = train_test_split(window_df[window_df.columns[0:window_size]], labels, test_size=0.2)

print(train.shape, test.shape)
print(train_labels.value_counts())
print(test_labels.value_counts())

# Timing training and testing.
rf_start = datetime.datetime.now()

# Build model and fit on training data.
rf_model = build_model(train, train_labels)
train_time = datetime.datetime.now() - rf_start

# Test.
rf_predictions = rf_model.predict(test)
test_time = datetime.datetime.now() - rf_start - train_time
rf_probs = rf_model.predict_proba(test)[:, 1]

print(train_time)
print(test_time)

# Generate and print out classification report.
report = classification_report(test_labels, rf_predictions)
accuracy = accuracy_score(test_labels, rf_predictions)*100
print('Classification Report:\n', report)

# Generate confusion matrix from results and calculate F1 score.
cm = confusion_matrix(test_labels, rf_predictions)
f_score = 2*cm[1][1] / (2*cm[1][1] + cm[1][0] + cm[0][1])
print(f_score)

# =============================================================================

# Alternatively, using PCs as input to model.
train = pc_df[pc_df.columns[0:window_size]]
train_labels = pc_df.Classification
# train, test, train_labels, test_labels = train_test_split(pc_df[pc_df.columns[0:window_size]], labels, test_size=0.2)

print(train.shape, test.shape)
print(train_labels.value_counts())
print(test_labels.value_counts())

# Timing training and testing.
rf_start = datetime.datetime.now()

# Build model and fit on training data.
rf_model = build_model(train, train_labels)
train_time = datetime.datetime.now() - rf_start

# Test.
rf_predictions = rf_model.predict(test)
test_time = datetime.datetime.now() - rf_start - train_time
rf_probs = rf_model.predict_proba(test)[:, 1]

print(train_time)
print(test_time)

# Generate and print out classification report.
report = classification_report(test_labels, rf_predictions)
accuracy = accuracy_score(test_labels, rf_predictions)*100
print('Classification Report:\n', report)

# Generate confusion matrix from results and calculate F1 score.
cm = confusion_matrix(test_labels, rf_predictions)
f_score = 2*cm[1][1] / (2*cm[1][1] + cm[1][0] + cm[0][1])
print(f_score)
