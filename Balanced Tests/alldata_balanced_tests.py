#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 08:13:05 2020

Replicating the balanced test conditions of the RF paper.

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from functions import *

# List of all houses w/ non-null EV power data.
house_ids = ['27', '661', '1222', '1642', '3000', '4373', '4767', '5679', '6139', '8156', '9053']

# Define window size, overlap and sampling interval.
window_size = 10
overlap = 5
sample_interval = 1

# window_size = 5
# overlap = 4
# sample_interval = 15

# List to store Dataframes for each houses data.
all_windows = []

total_windows = 0
# Read in each houses data into Dataframes from csv files.
for x in house_ids:
    
    window_df, house_df = csv_data_to_windows(x, window_size, overlap, sample_interval)
    total_windows += len(window_df)
    
    # Select equal number of Charging & Disconnected windows from current house.
    # Balance by undersampling.
    window_df = undersample(window_df)
    
    # Add Dataframe to list.
    window_df['Data ID'] = int(x)
    all_windows.append(window_df)
    
# Combine all data into one Dataframe.
window_df = pd.concat(all_windows, ignore_index = True)
window_df.reset_index(drop=True, inplace=True)
# Drop null rows.
window_df = window_df.dropna()
print(total_windows)
print(len(window_df))

class_counts = window_df.Classification.value_counts()
print(class_counts)

# Split data into train & test sets (80:20) using windowed active power directly for input to  model.
# =============================================================================

labels = window_df.Classification
train, test, train_labels, test_labels = train_test_split(window_df[window_df.columns[0:window_size]], labels, test_size=0.2)

print(train.shape, test.shape)
print(train_labels.value_counts())
print(test_labels.value_counts())

# Timing training and testing.
rf_start = datetime.datetime.now()

# Build model and fit on training data.
rf_model = build_model(train, train_labels, model_name='RF')
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


# Alternatively, using PCs as input to model.
# =============================================================================

train_df = pd.DataFrame().from_records(train)
train_df['Classification'] = train_labels
pc_df = apply_pca(train_df, window_size)
train = pc_df[pc_df.columns[0:window_size]]

test_df = pd.DataFrame().from_records(test)
test_df['Classification'] = test_labels
pc_df = apply_pca(test_df, window_size)
test = pc_df[pc_df.columns[0:window_size]]

# train, test, train_labels, test_labels = train_test_split(pc_df[pc_df.columns[0:window_size]], labels, test_size=0.2)

print(train.shape, test.shape)
print(train_labels.value_counts())
print(test_labels.value_counts())

# Timing training and testing.
rf_start = datetime.datetime.now()

# Build model and fit on training data.
rf_model = build_model(train, train_labels, model_name='RF')
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
cm1 = confusion_matrix(test_labels, rf_predictions)
f_score = 2*cm1[1][1] / (2*cm1[1][1] + cm1[1][0] + cm1[0][1])
print(f_score)


# Trying other feature extraction techniques such as LDA.
# =============================================================================

# train_df = pd.DataFrame().from_records(train)
# train_df['Classification'] = train_labels
# standard = StandardScaler().fit_transform(train)
# lda = LDA(n_components=1)
# train = lda.fit_transform(standard, train_labels)
# test = lda.transform(test)

# # train, test, train_labels, test_labels = train_test_split(pc_df[pc_df.columns[0:window_size]], labels, test_size=0.2)

# print(train.shape, test.shape)
# print(train_labels.value_counts())
# print(test_labels.value_counts())

# # Timing training and testing.
# rf_start = datetime.datetime.now()

# # Build model and fit on training data.
# rf_model = build_model(train, train_labels)
# train_time = datetime.datetime.now() - rf_start

# # Test.
# rf_predictions = rf_model.predict(test)
# test_time = datetime.datetime.now() - rf_start - train_time
# rf_probs = rf_model.predict_proba(test)[:, 1]

# print(train_time)
# print(test_time)

# # Generate and print out classification report.
# report = classification_report(test_labels, rf_predictions)
# accuracy = accuracy_score(test_labels, rf_predictions)*100
# print('Classification Report:\n', report)

# # Generate confusion matrix from results and calculate F1 score.
# cm2 = confusion_matrix(test_labels, rf_predictions)
# f_score = 2*cm2[1][1] / (2*cm2[1][1] + cm2[1][0] + cm2[0][1])
# print(f_score)